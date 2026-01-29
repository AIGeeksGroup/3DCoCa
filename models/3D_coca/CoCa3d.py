import math, os
from functools import partial

import numpy as np
import torch
from torch import nn, Tensor
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import furthest_point_sample

from utils.misc import huber_loss
from utils.pc_util import scale_points, shift_scale_points
from datasets.scannet import BASE
from typing import Dict

from .config import model_config
from .criterion import build_criterion
from .helpers import GenericMLP

from .vote_query import VoteQuery

from .position_embedding import PositionEmbeddingCoordsSine

from .transformer import (
    MaskedTransformerEncoder, TransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder,
    TransformerEncoderLayer)
from typing import Optional

import MinkowskiEngine as ME
from .epcl_detection import build_epcl

from .coca_pytorch import CoCa

class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


class CoCa_3D(nn.Module):
    
    def __init__(
        self,
        config,
        dataset_config,
        train_dataset,
        decoder,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
        criterion=None,
        voxel_size=0.02
    ):
        super().__init__()
        self.decoder = decoder
        self.num_queries = num_queries
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.voxel_size = voxel_size
        self.dataset_config = dataset_config
        self.tokenizer = train_dataset.tokenizer
        self.nvocabs = len(self.tokenizer)

        self.epcl = build_epcl(
            type('args', (object,), {
                'PREENC_NPOINTS': 512,
                'GROUPS_SIZE': 16,
                'ENC_DIM': encoder_dim,
                'PREENCODER_DIM': 512,  # Modified to match actual input feature dimension
                'VOXEL_SIZE': voxel_size
            })
        )

        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )

        self.coca = CoCa(
            dim=config.text_dim,
            image_dim=config.image_dim,
            num_tokens= self.nvocabs,
            multimodal_depth=config.multimodal_depth,
            unimodal_depth=config.text_encoder_depth,
            heads=config.text_heads,
            ff_mult=config.text_ff_mult,
        )

        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.vote_query_generator = VoteQuery(decoder_dim, num_queries)
        
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        
        self.revote_layers = [0, 1, 2]
        self.revoting_module = nn.ModuleDict({
            f'layer-{layer_id}': nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim),
                nn.LayerNorm(decoder_dim),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(decoder_dim, 3)
            ) for layer_id in self.revote_layers
        })
        
        self.interim_proj = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(decoder_dim),
                    nn.Linear(decoder_dim, decoder_dim),
                    nn.LayerNorm(decoder_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(decoder_dim, decoder_dim),
                    nn.Dropout(p=0.3)
                ) for _ in self.decoder.layers
            ])
        
        self.mlp_heads = self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)
        
        self.box_processor = BoxProcessor(dataset_config)
        self.criterion = criterion

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        return nn.ModuleDict(mlp_heads)
    
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].contiguous()
            if pc.size(-1) > 6
            else None
        )
        return xyz, features
    def fps_downsample(self,out_coords, out_feats, batch_indices, batch_size, num_sample=1024):

        device = out_coords.device
        feature_dim = out_feats.shape[-1]

        enc_xyz = torch.zeros((batch_size, num_sample, 3), device=device)
        enc_features = torch.zeros((batch_size, num_sample, feature_dim), device=device)

        for b in range(batch_size):
            batch_mask = (batch_indices == b)
            coords_b = out_coords[batch_mask]   # [valid_points, 3]
            feats_b  = out_feats[batch_mask]    # [valid_points, C]
            valid_points = coords_b.shape[0]

            if valid_points == 0:
                continue

            coords_b_expand = coords_b.unsqueeze(0).contiguous()

            if valid_points >= num_sample:
                fps_idx = furthest_point_sample(coords_b_expand, num_sample)  # [1, num_samples]
            else:
                # valid_points < num_samples
                fps_idx = furthest_point_sample(coords_b_expand, valid_points)  # [1, valid_points]
                pad_len = num_sample - valid_points
                if pad_len > 0:
                    tail = torch.full((1, pad_len),
                                    valid_points - 1,
                                    dtype=torch.long, device=device)
                    fps_idx = torch.cat([fps_idx, tail], dim=1)  # [1, num_samples]

            fps_idx = fps_idx.clamp(0, valid_points - 1)

            # coords_b: [valid_points, 3], feats_b: [valid_points, C]
            fps_idx= fps_idx.long()
            coords_sampled = torch.gather(
                coords_b, 
                dim=0,
                index=fps_idx.view(-1,1).expand(-1, coords_b.shape[1])
            )  # => [num_samples, 3]

            feats_sampled = torch.gather(
                feats_b,
                dim=0,
                index=fps_idx.view(-1,1).expand(-1, feats_b.shape[1])
            )  # => [num_samples, C]

            enc_xyz[b]      = coords_sampled
            enc_features[b] = feats_sampled
        return enc_xyz, enc_features,fps_idx
    
    def decode_coca_logits(
        self,
        text_logits: torch.Tensor,  # [batch, seq_len, vocab_size]
        tokenizer,                  # e.g. ScanReferTokenizer
        beam_search: bool = False,
        beam_size: int = 3
    ):
        import torch.nn.functional as F
        
        B, seq_len, vocab_size = text_logits.shape
        if not beam_search:
            # Greedy
            pred_ids = text_logits.argmax(dim=-1).cpu().numpy()  # [B, seq_len]
            captions = []
            for row_ids in pred_ids:
                # decode
                cap_str = tokenizer.decode(row_ids.tolist())
                captions.append(cap_str)
            return captions
        else:
            # Simple beam search example
            all_captions = []
            for b_idx in range(B):
                single_logits = text_logits[b_idx]  # [seq_len, vocab_size]
                beams = [([], 0.0)]
                for t in range(seq_len):
                    next_beams = []
                    prob_t = F.log_softmax(single_logits[t], dim=-1)  # [vocab_size]
                    topk = torch.topk(prob_t, beam_size)
                    for (tokens_so_far, logp_so_far) in beams:
                        for i in range(beam_size):
                            token_id = topk.indices[i].item()
                            token_logp = topk.values[i].item()
                            new_tokens = tokens_so_far + [token_id]
                            new_logp   = logp_so_far + token_logp
                            next_beams.append((new_tokens, new_logp))
                    next_beams.sort(key=lambda x: x[1], reverse=True)
                    beams = next_beams[:beam_size]
                # best
                best_tokens, _ = beams[0]
                cap_str = tokenizer.decode(best_tokens)
                all_captions.append(cap_str)
            return all_captions
    def run_encoder(self, point_clouds):
        """
        Enhanced encoder function with robust feature handling
        """
        batch_size = point_clouds.shape[0]
        device = point_clouds.device
        
        # Extract coordinates and features
        xyz, features = self._break_up_pc(point_clouds)
        B,N,_ = xyz.shape
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N).unsqueeze(-1)
        coordinates = torch.cat([batch_idx.float(), xyz], dim=-1)
        coordinates = coordinates.view(-1, coordinates.shape[-1])
        if features is not None:
            batch_features = features.view(-1, features.shape[-1])
        else:
            batch_features = None
        
        coords_xyz = coordinates[:, 1:]  # [B*N, 3]
        coords_xyz = torch.div(coords_xyz, self.voxel_size, rounding_mode='floor').int()
        new_coordinates = torch.cat([coordinates[:, 0:1].int(), coords_xyz], dim=1)  # [B*N, 4]
        torch.cuda.synchronize(device=device)

        sp_tensor = ME.SparseTensor(
            features=batch_features.float().contiguous(),
            coordinates=new_coordinates.contiguous(),
            device=device,
            coordinate_manager=None
        )
        
        # Forward pass through modified EPCL model that now handles 7D input properly
        epcl_output = self.epcl({'sp_tensor': sp_tensor})
        
        output_sp = epcl_output['sp_tensor']
        
        out_coords = output_sp.C[:, 1:4].float()
        out_feats = output_sp.F
        
        batch_indices = output_sp.C[:, 0].long()
        num_samples = self.num_queries
        feature_dim = out_feats.shape[-1]
        enc_xyz = torch.zeros((batch_size, num_samples, 3), device=device)
        enc_features = torch.zeros((batch_size, num_samples, feature_dim), device=device)
        fps_indices = []
        for b in range(batch_size):
            batch_mask = batch_indices == b
            valid_points = batch_mask.sum().item()
            if valid_points == 0:
                indices = torch.full((1, num_samples), -1, dtype=torch.long, device=device)
                fps_indices.append(indices)
                continue
            batch_coords = out_coords[batch_mask].unsqueeze(0).contiguous()
            feats_b  = out_feats[batch_mask].contiguous()
            if valid_points >= num_samples:
                indices = furthest_point_sample(batch_coords, num_samples)
            else:

                indices = furthest_point_sample(batch_coords, valid_points)
                padding = torch.full((1, num_samples - valid_points), 
                                   valid_points-1,
                                   dtype=torch.long, device=device)
                indices = torch.cat([indices, padding], dim=1)
            indices = indices.clamp(0, valid_points-1)
            fps_indices.append(indices)
            coords_b_2d = batch_coords[0]
            idx_expand = indices.view(-1,1).expand(-1,3)
            coords_sampled = torch.gather(coords_b_2d,0,idx_expand.long())
            enc_xyz[b] = coords_sampled

            idx_expand_feat = indices.view(-1,1).expand(-1,feature_dim)
            feats_sampled = torch.gather(feats_b,0,idx_expand_feat.long())
            enc_features[b] = feats_sampled
        fps_indices = torch.cat(fps_indices, dim=0)
        enc_features_decoder = enc_features.permute(1, 0, 2).contiguous()  # [num_samples, batch_size, channel]

        return enc_xyz, enc_features_decoder, fps_indices

    def run_decoder(self, tgt, memory, enc_xyz, query_xyz, query_outputs, input_range):
        
        # batch x channel x npenc
        enc_pos = self.pos_embedding(enc_xyz, input_range=input_range)
        enc_pos = enc_pos.permute(2, 0, 1)
        
        output = tgt.repeat(2, 1, 1)
        
        intermediate = []
        attns = []
        layer_query_xyz = [query_xyz]
        
        tgt_mask = torch.zeros((output.shape[0], output.shape[0]), device=output.device)
        tgt_mask[:self.num_queries, self.num_queries:] = 1
        tgt_mask[self.num_queries:, :self.num_queries] = 1
        
        tgt_mask = tgt_mask.bool()
        
        for dec_layer_id, layer in enumerate(self.decoder.layers):
            
            query_pos = self.pos_embedding(query_xyz, input_range=input_range)
            query_pos = self.query_projection(query_pos)
            query_pos = query_pos.permute(2, 0, 1)
            
            query_pos = query_pos.repeat(2, 1, 1)
            
            output, attn = layer(
                output, 
                memory, 
                pos=enc_pos, 
                query_pos=query_pos, 
                tgt_mask=tgt_mask,
                return_attn_weights=True
            )
            
            # interaction
            intermediate.append(self.decoder.norm(output))
            attns.append(attn)
            
            # ==== revote to object center: 
            #   ntoken x batch x channel -> batch x ntoken x channel
            if dec_layer_id in self.revote_layers:
                step_shift = self.revoting_module[f'layer-{dec_layer_id}'](
                    intermediate[-1][:self.num_queries].permute(1, 0, 2)
                )
                query_xyz = query_xyz + 0.2 * torch.sigmoid(step_shift) - 0.1
            
            layer_query_xyz.append(query_xyz)

        attns = torch.stack(attns)
        intermediate = torch.stack(intermediate)
        layer_query_xyz = torch.stack(layer_query_xyz)
        
        return layer_query_xyz, intermediate, attns


    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: num_layers x batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz[l], point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs,is_eval=False):
        
        point_clouds = inputs["point_clouds"]
        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        text_tokens = inputs["reference_tokens"]
        batch_size = point_clouds.shape[0]
        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds) # [num_samples,batch, channel]

        enc_features = enc_features.permute(1, 2, 0)  # [batch, channel, num_samples]
        # batch_size, num_samples, channel = enc_features_reshaped.shape
        query_outputs = self.vote_query_generator(
            enc_xyz, enc_features
            )
        query_outputs['seed_inds'] = enc_inds
        query_xyz = query_outputs['query_xyz']
        query_features = query_outputs["query_features"]
        # enc_features_proj = torch.stack(proj_features, dim=1)  # [num_samples, batch, channel]
        enc_features = self.encoder_to_decoder_projection(enc_features)# [batch, channel, num_samples]
        enc_features = enc_features.permute(2, 0, 1)# [num_samples, batch, channel]
        
        tgt = query_features.permute(2, 0, 1)  # [nqueries, batch, channel]
        
        layer_query_xyz, box_features, attn = self.run_decoder(
            tgt, enc_features, enc_xyz, query_xyz, query_outputs, input_range=point_cloud_dims
        )   # nlayers x nqueries x batch x channel
        
        sample_inds = torch.clamp(query_outputs['sample_inds'].long(), 0, enc_inds.shape[1]-1)
        query_outputs['revote_seed_inds'] = torch.gather(enc_inds, 1, query_outputs['sample_inds'].long())
        query_outputs['revote_seed_xyz'] = \
            torch.gather(enc_xyz, dim=1, index=query_outputs['sample_inds'].long()[..., None].repeat(1, 1, 3))
        query_outputs['revote_vote_xyz'] = layer_query_xyz[1:]  # nlayers x batch x nqueries x 3
        
        box_predictions = self.get_box_predictions(
            layer_query_xyz, point_cloud_dims, box_features[:, :self.num_queries]
        )
        text_tokens = text_tokens.view(-1,text_tokens.shape[-1])
        image_tokens = enc_features.reshape(-1,enc_features.shape[-1])
        if not is_eval:
            total_coca_loss = self.coca(
                text=text_tokens,
                image_tokens=image_tokens,  # [B, N, dec_dim]
                labels=inputs.get("caption_labels", None),  # e.g. if you do self-shift for LM
                return_loss=True
            )

        else:
            # print(f"输入的文本形状: {text_tokens.shape}")
            # print(f"输入的图像形状: {image_tokens.shape}")
            text_logits_or_embeds = self.coca(
                text=text_tokens,
                image_tokens=image_tokens,
                return_loss=False
            )
            # print(f"文本 logits 或 embeddings 形状: {text_logits_or_embeds.shape}")
            pred_captions = self.decode_coca_logits(
                text_logits=text_logits_or_embeds,
                tokenizer=self.tokenizer,
                beam_search=False,
                beam_size=3
            )
            num_decoded = len(pred_captions)
            # print(f"解码得到的句子数量: {num_decoded}")
            batch_size, nqueries_expected, _, _ = box_predictions['outputs']["box_corners"].shape
            if num_decoded == batch_size * nqueries_expected:
                nqueries = nqueries_expected
            elif num_decoded == batch_size:
                nqueries = 1
            
            idx = 0
            lang_cap = []
            for b in range(batch_size):
                row_caps = []
                for q in range(self.num_queries):
                    cap_str = pred_captions[idx]
                    idx += 1
                    cap_str = f'sos {cap_str} eos'
                    row_caps.append(cap_str)
                lang_cap.append(row_caps)
            box_predictions['outputs']['lang_cap'] = lang_cap

        if not is_eval and (self.criterion is not None):
            assign_res, detection_loss, _ = self.criterion(query_outputs, box_predictions, inputs)
            total_loss = detection_loss + total_coca_loss

            box_predictions['outputs']['loss'] = total_loss
            box_predictions['outputs']['assignments'] = assign_res

        if not is_eval:
            return {
                "box_outputs": box_predictions["outputs"],  # dict, include final boxes
                "coca_loss": total_coca_loss
            }
        else:
            return {
                "box_outputs": box_predictions["outputs"],
                "text_logits": text_logits_or_embeds
            }



def build_preencoder(cfg):
    mlp_dims = [cfg.in_channel, 64, 128, cfg.enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=cfg.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_decoder(cfg):
    decoder_layer = TransformerDecoderLayer(
        d_model=cfg.dec_dim,
        nhead=cfg.dec_nhead,
        dim_feedforward=cfg.dec_ffn_dim,
        dropout=cfg.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=cfg.dec_nlayers, return_intermediate=True
    )
    return decoder


def detector(args, dataset_config,train_dataset):
    cfg = model_config(args, dataset_config)
    
    decoder = build_decoder(cfg)
    
    criterion = build_criterion(cfg, dataset_config)
    
    model = CoCa_3D(
        cfg,
        cfg.dataset_config,
        train_dataset,
        decoder,
        encoder_dim=cfg.enc_dim,
        decoder_dim=cfg.dec_dim,
        mlp_dropout=cfg.mlp_dropout,
        num_queries=cfg.nqueries,
        criterion=criterion,
        voxel_size=0.02
    )
    return model
