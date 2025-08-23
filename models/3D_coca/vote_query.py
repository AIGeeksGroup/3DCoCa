import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.pointnet2.pointnet2_modules import (
    PointnetSAModuleVotes
)
import third_party.pointnet2.pointnet2_utils as pointnet2_utils


class VoteQuery(nn.Module):
    def __init__(self, d_model, nqueries):
        super().__init__()
        self.nqueries = nqueries

        self.FFN_vote = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, 3 + d_model, 1),
        )

        self.set_abstraction = PointnetSAModuleVotes(
            npoint=nqueries,
            radius=0.3,
            nsample=16,
            mlp=[d_model, d_model, d_model, d_model],
            use_xyz=True,
            normalize_xyz=True
        )

    def forward(self, encode_xyz, encode_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """

        # batch, channel, npoints
        encode_features = encode_features.contiguous()
        out = self.FFN_vote(encode_features)

        coord_shift = torch.clamp(torch.sigmoid(out[:, :3, :]), 0.1, 0.9)  # 限制梯度范围
        vote_xyz = encode_xyz.contiguous() + coord_shift.permute(0, 2, 1).contiguous() - 0.5

        encode_features = encode_features + out[:, 3:, :]

        features_norm = torch.norm(encode_features, p=2, dim=1)
        encode_features = encode_features.div(features_norm.unsqueeze(1))

        sample_inds = pointnet2_utils.furthest_point_sample(
            encode_xyz, self.nqueries
        )

        query_xyz, query_features, _ = self.set_abstraction(
            vote_xyz,
            encode_features,
            sample_inds
        )

        return {
            'vote_xyz': vote_xyz,  # batch x npenc x 3
            'seed_xyz': encode_xyz,  # batch x npenc x 3
            'sample_inds': sample_inds,  # batch x nquery x 3
            'query_xyz': query_xyz,  # batch x nquery x 3
            'query_features': query_features  # batch x channel x nquery
        }

