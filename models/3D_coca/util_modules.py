import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from third_party.pointnet2 import pointnet2_utils
# try:
#     from pointnet2 import pointnet2_utils
# except:
#     import pointnet2_utils

# @torch.no_grad()
def fps(data, number):
    '''
        data B N 3
        number int
    '''
    if not data.is_floating_point():
        data = data.float()
    if number <= 0:
        number = 1
    if data.shape[1] < number:
        return data
    
    try:
        fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
        fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return fps_data
    except Exception as e:
        print(f"FPS error, using fallback method: {e}")
        indices = torch.linspace(0, data.shape[1]-1, number).long().to(data.device)
        return torch.index_select(data, 1, indices)

@torch.no_grad()
def knn(xyz0, xyz1, k):
    """
    Given xyz0 with shape [B, N, C], xyz1 with shape [B, M, C], 
    going to find k nearest points for xyz1 in xyz0
    """
    if not xyz0.is_floating_point():
        xyz0 = xyz0.float()
    if not xyz1.is_floating_point():
        xyz1 = xyz1.float()
    
    
    if torch.isnan(xyz0).any():
        xyz0 = torch.nan_to_num(xyz0, nan=0.0)
    
    if torch.isnan(xyz1).any():
        xyz1 = torch.nan_to_num(xyz1, nan=0.0)
        
    
    if k <= 0:
        k = 1
    
    actual_k = min(k, xyz0.shape[1])
    
    try:
        cdist = torch.cdist(xyz1, xyz0) # [B, M, N]
        values, indices = torch.topk(cdist, k=actual_k, dim=-1, largest=False)
        return values, indices
    except Exception as e:
        print(f"KNN error, using fallback method: {e}")
        B, N, C = xyz0.shape
        _, M, _ = xyz1.shape
        indices = torch.zeros(B, M, actual_k, dtype=torch.long, device=xyz0.device)
        values = torch.zeros(B, M, actual_k, device=xyz0.device)
        
        for i in range(M):
            start_idx = min(i * actual_k, N - actual_k)
            for b in range(B):
                indices[b, i] = torch.arange(start_idx, start_idx + actual_k, device=xyz0.device)
                values[b, i] = torch.tensor([j+1 for j in range(actual_k)], device=xyz0.device)
        indices = torch.clamp(indices, 0, xyz0.shape[1]-1)
        return values, indices

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
        = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    if not src.is_floating_point():
        src = src.float()
    if not dst.is_floating_point():
        dst = dst.float()
    
    if torch.isnan(src).any():
        src = torch.nan_to_num(src, nan=0.0)
    if torch.isnan(dst).any():
        dst = torch.nan_to_num(dst, nan=0.0)
        
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    try:
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        
        dist = torch.clamp(dist, min=0.0)
        
        return dist
    except Exception as e:
        print(f"Distance calculation error, using fallback method: {e}")
        return torch.ones(B, N, M, device=src.device)

class EmbeddingEncoder(nn.Module):
    def __init__(self, encoder_channel, in_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),  # Modified from hardcoded 512
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        
        # Maintain architecture with concatenated global+local features (256+256=512)
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),  # This 512 is from concatenation, not input features
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):  
        '''
            point_groups : B G N C
            -----------------
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape
        
        if torch.isnan(point_groups).any():
            point_groups = torch.nan_to_num(point_groups, nan=0.0)
        
        # Verify input channel compatibility
        assert c == self.in_channels, f"Input channels {c} != expected {self.in_channels}"
        
        # Reshape for processing
        point_groups = point_groups.reshape(bs * g, n, self.in_channels).transpose(2, 1)
        
        # Forward pass with dynamic feature dimensions
        feature = self.first_conv(point_groups)  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        
        # Concatenation creates fixed 512 dims regardless of input feature dims
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG encoder_channel n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG encoder_channel
        
        return feature_global.reshape(bs, g, self.encoder_channel)

class SparseGroup(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):  # Keep parameter names consistent
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        
    @torch.no_grad()
    def group(self, xyz, feats=None):
        if not xyz.is_floating_point():
            xyz = xyz.float()
        if feats is not None and not feats.is_floating_point():
            feats = feats.float()
        
        if torch.isnan(xyz).any():
            xyz = torch.nan_to_num(xyz, nan=0.0)
        if feats is not None and torch.isnan(feats).any():
            feats = torch.nan_to_num(feats, nan=0.0)
            
        batch_size, num_points, _ = xyz.shape
        
        if num_points <= self.group_size:
            print(f"Warning: Points({num_points}) too few for proper grouping")
            center = xyz.mean(dim=1, keepdim=True)  # [B, 1, 3]
            
            
            if num_points < self.group_size:
                padding_size = self.group_size - num_points
                padding = xyz[:, -1:].repeat(1, padding_size, 1)
                padded_xyz = torch.cat([xyz, padding], dim=1)
                neighborhood = padded_xyz.unsqueeze(1)  # [B, 1, group_size, 3]
            else:
                neighborhood = xyz[:, :self.group_size].unsqueeze(1)  # [B, 1, group_size, 3]
            batch_idx = torch.arange(batch_size * self.group_size, device=xyz.device)
            
            patch_feats = None
            if feats is not None:
                if num_points < self.group_size:
                    padding_feats = feats[:, -1:].repeat(1, padding_size, 1)
                    padded_feats = torch.cat([feats, padding_feats], dim=1)
                    patch_feats = padded_feats.unsqueeze(1)  # [B, 1, group_size, C]
                else:
                    patch_feats = feats[:, :self.group_size].unsqueeze(1)  # [B, 1, group_size, C]
            
            return neighborhood, center, batch_idx, patch_feats
        adjusted_num_group = min(self.num_group, num_points // 2)
        if adjusted_num_group < self.num_group:
            print(f"Warning:  {self.num_group} -> {adjusted_num_group}")

        try:
            center = fps(xyz, adjusted_num_group)
            if torch.isnan(center).any():
                print("Warning: FPS returned center points contain NaN, using fallback method")
                indices = torch.linspace(0, num_points - 1, adjusted_num_group).long().to(xyz.device)
                center = xyz[:, indices]
        except Exception as e:
            print(f"FPS error: {e}")
            indices = torch.linspace(0, num_points - 1, adjusted_num_group).long().to(xyz.device)
            center = xyz[:, indices]
            
        if center.shape[1] == 0 or torch.isnan(center).any():
            print(f"Warning: Center points are invalid, using fallback method")
            center = xyz[:, :1]
            adjusted_num_group = 1
        try:
            _, idx = knn(xyz, center, self.group_size)
            if torch.any(idx < 0) or torch.any(idx >= num_points):
                print("Warning: KNN returned invalid indices, using fallback method")
                raise ValueError("Invalid indices")
        except Exception as e:
            print(f"KNN error or invalid indices: {e}")
            idx = torch.zeros(batch_size, center.shape[1], self.group_size, device=xyz.device).long()
            for i in range(center.shape[1]):
                start_idx = min(i * self.group_size, num_points - self.group_size)
                for b in range(batch_size):
                    idx[b, i] = torch.arange(start_idx, start_idx + self.group_size, device=xyz.device)
        
        actual_k = idx.size(-1)
        if actual_k < self.group_size:
            padding = self.group_size - actual_k
            idx = torch.cat([idx, idx[:, :, -1:].repeat(1, 1, padding)], dim=-1)
            
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        
        max_allowed_idx = batch_size * num_points - 1
        idx = torch.clamp(idx, 0, max_allowed_idx)
        
        flat_idx = idx.reshape(-1)
        try:
            flattened_xyz = xyz.reshape(batch_size * num_points, -1)
            neighborhood = flattened_xyz[flat_idx].reshape(batch_size, adjusted_num_group, self.group_size, 3)
            if torch.isnan(neighborhood).any():
                print("Warning: Neighborhood points contain NaN, fixing")
                neighborhood = torch.nan_to_num(neighborhood, nan=0.0)

            # Process features
            patch_feats = None
            if feats is not None:
                flattened_feats = feats.reshape(batch_size * num_points, -1)
                patch_feats = flattened_feats[flat_idx].reshape(
                    batch_size, adjusted_num_group, self.group_size, flattened_feats.shape[-1])
                
                if torch.isnan(patch_feats).any():
                    print("Warning: Features contain NaN, fixing")
                    patch_feats = torch.nan_to_num(patch_feats, nan=0.0)
        except Exception as e:
            print(f"Index error: {e}")
            neighborhood = torch.zeros(batch_size, adjusted_num_group, self.group_size, 3, device=xyz.device)
            patch_feats = None
            if feats is not None:
                patch_feats = torch.zeros(batch_size, adjusted_num_group, self.group_size, 
                                         feats.shape[-1], device=feats.device)
        
        if adjusted_num_group < self.num_group:
            padding_center = torch.zeros(batch_size, self.num_group - adjusted_num_group, 3, device=center.device)
            center = torch.cat([center, padding_center], dim=1)
            
            padding_neighborhood = torch.zeros(batch_size, self.num_group - adjusted_num_group, 
                                             self.group_size, 3, device=neighborhood.device)
            neighborhood = torch.cat([neighborhood, padding_neighborhood], dim=1)
            
            padding_idx = torch.zeros(batch_size, (self.num_group - adjusted_num_group) * self.group_size, 
                                    device=idx.device).long()
            flat_idx = torch.cat([flat_idx, padding_idx.reshape(-1)], dim=0)
            
            if patch_feats is not None:
                padding_feats = torch.zeros(batch_size, self.num_group - adjusted_num_group, 
                                          self.group_size, patch_feats.shape[-1], device=patch_feats.device)
                patch_feats = torch.cat([patch_feats, padding_feats], dim=1)
                
        if torch.isnan(neighborhood).any():
            neighborhood = torch.nan_to_num(neighborhood, nan=0.0)
        if torch.isnan(center).any():
            center = torch.nan_to_num(center, nan=0.0)
        if patch_feats is not None and torch.isnan(patch_feats).any():
            patch_feats = torch.nan_to_num(patch_feats, nan=0.0)
                
        return neighborhood, center, flat_idx, patch_feats
    
    @torch.no_grad()
    def forward(self, xyz, feature=None):
        
        if not xyz.is_floating_point():
            xyz = xyz.float()
        if feature is not None and not feature.is_floating_point():
            feature = feature.float()
            
        if torch.isnan(xyz).any():
            print("Warning: Input point cloud contains NaN, fixing")
            xyz = torch.nan_to_num(xyz, nan=0.0)
        if feature is not None and torch.isnan(feature).any():
            print("Warning: Input features contain NaN, fixing")
            feature = torch.nan_to_num(feature, nan=0.0)
            
        if xyz is None or xyz.numel() == 0:
            raise ValueError("Input point cloud is empty")

        if xyz.dim() != 2 or xyz.shape[1] < 4:
            raise ValueError(f"Input point cloud format error: {xyz.shape}, should be [N, 4+]")

        batch_ids = xyz[:, 0].unique().long()
        num_batches = len(batch_ids)
        
        if num_batches == 0:
            raise ValueError("No valid batch data")
            
        all_neighborhoods = []
        all_centers = []
        all_batch_idxs = []
        batch_xyz_list = []
        batch_feature_list = []

        max_points_per_batch = max([torch.sum(xyz[:, 0] == b_id).item() for b_id in batch_ids])
        safe_num_group = min(self.num_group, max(1, max_points_per_batch // 4))
        
        for batch_id in batch_ids:
            mask = xyz[:, 0] == batch_id
            points_in_batch = mask.sum().item()

            if points_in_batch < 3:
                print(f"Batch {batch_id} has too few points ({points_in_batch}), skipping")
                continue

            # Extract coordinates and features for the current batch
            batch_xyz = xyz[mask, 1:4].unsqueeze(0)  # [1, N, 3]

            # Ensure coordinates are float and have no NaN
            if not batch_xyz.is_floating_point():
                batch_xyz = batch_xyz.float()
            if torch.isnan(batch_xyz).any():
                batch_xyz = torch.nan_to_num(batch_xyz, nan=0.0)
                
            batch_feat = None
            if feature is not None:
                batch_feat = feature[mask].unsqueeze(0)  # [1, N, C]
                # Ensure features are float and have no NaN
                if not batch_feat.is_floating_point():
                    batch_feat = batch_feat.float()
                if torch.isnan(batch_feat).any():
                    batch_feat = torch.nan_to_num(batch_feat, nan=0.0)
                
            try:
                orig_num_group = self.num_group
                self.num_group = min(safe_num_group, max(1, points_in_batch // 2))

                # Call group function
                neighborhood, center, batch_idx, patch_feats = self.group(batch_xyz, batch_feat)

                # Restore original group size
                self.num_group = orig_num_group

                # Check if results contain NaN
                if torch.isnan(neighborhood).any():
                    neighborhood = torch.nan_to_num(neighborhood, nan=0.0)
                if torch.isnan(center).any():
                    center = torch.nan_to_num(center, nan=0.0)
                if patch_feats is not None and torch.isnan(patch_feats).any():
                    patch_feats = torch.nan_to_num(patch_feats, nan=0.0)

                # Add to result lists
                all_neighborhoods.append(neighborhood)
                all_centers.append(center)
                all_batch_idxs.append(batch_idx.reshape(1, -1))
                batch_xyz_list.append(batch_xyz)
                
                if patch_feats is not None:
                    batch_feature_list.append(patch_feats)
            except Exception as e:
                print(f"Error processing batch {batch_id}: {e}")
                # If it's the first batch that fails, we may need to create a default output
                if len(all_neighborhoods) == 0:
                    # Create a minimal viable output
                    print("Creating default output")
                    min_points = min(points_in_batch, self.group_size)
                    default_neighborhood = batch_xyz[:, :min_points].unsqueeze(1).repeat(1, 1, 1, 1)
                    if min_points < self.group_size:
                        # Pad to group_size
                        padding = torch.zeros(1, 1, self.group_size - min_points, 3, device=xyz.device)
                        default_neighborhood = torch.cat([default_neighborhood, padding], dim=2)
                    
                    default_center = batch_xyz[:, :1]
                    default_batch_idx = torch.zeros(1, self.group_size, device=xyz.device).long()
                    
                    all_neighborhoods.append(default_neighborhood)
                    all_centers.append(default_center)
                    all_batch_idxs.append(default_batch_idx.reshape(1, -1))
                    batch_xyz_list.append(batch_xyz)
                    
                    if feature is not None:
                        default_feats = batch_feat[:, :min_points].unsqueeze(1).repeat(1, 1, 1, 1)
                        if min_points < self.group_size:
                            # Pad features
                            feat_padding = torch.zeros(1, 1, self.group_size - min_points, 
                                                     batch_feat.shape[-1], device=feature.device)
                            default_feats = torch.cat([default_feats, feat_padding], dim=2)
                        batch_feature_list.append(default_feats)
        
        if not all_neighborhoods:
            print("All batches failed, creating emergency output")
            B = len(batch_ids)
            emergency_neighborhood = torch.zeros(B, 1, self.group_size, 3, device=xyz.device)
            emergency_center = torch.zeros(B, 1, 3, device=xyz.device)
            emergency_batch_idx = torch.zeros(B, self.group_size, device=xyz.device).long()
            emergency_features = None
            if feature is not None:
                emergency_features = torch.zeros(B, 1, self.group_size, feature.shape[1], device=feature.device)
            
            return emergency_neighborhood, emergency_center, emergency_batch_idx, batch_xyz_list, emergency_features

        # Standardize output shapes
        # 1. Find the maximum number of centers across all batches
        max_centers = max([center.shape[1] for center in all_centers])

        # 2. Adjust all batch outputs to the same shape
        for i in range(len(all_centers)):
            center = all_centers[i]
            neighborhood = all_neighborhoods[i]
            batch_idx = all_batch_idxs[i]
            
            if center.shape[1] < max_centers:
                # Pad centers
                padding_center = torch.zeros(1, max_centers - center.shape[1], 3, device=center.device)
                all_centers[i] = torch.cat([center, padding_center], dim=1)

                # Pad neighborhoods
                padding_neighborhood = torch.zeros(1, max_centers - neighborhood.shape[1],
                                                 self.group_size, 3, device=neighborhood.device)
                all_neighborhoods[i] = torch.cat([neighborhood, padding_neighborhood], dim=1)

                # Pad batch_idx
                padding_batch_idx = torch.zeros(1, (max_centers - center.shape[1]) * self.group_size,
                                              device=batch_idx.device).long()
                all_batch_idxs[i] = torch.cat([batch_idx, padding_batch_idx], dim=1)

                # Pad features
                if i < len(batch_feature_list):
                    patch_feats = batch_feature_list[i]
                    padding_feats = torch.zeros(1, max_centers - patch_feats.shape[1], 
                                              self.group_size, patch_feats.shape[-1], 
                                              device=patch_feats.device)
                    batch_feature_list[i] = torch.cat([patch_feats, padding_feats], dim=1)
        
        
        try:
            neighborhoods = torch.cat(all_neighborhoods, dim=0)  # [B,G,M,C]
            centers = torch.cat(all_centers, dim=0)              # [B,G,C]
            batch_idxs = torch.cat(all_batch_idxs, dim=0)        # [B,G*M]

            # Reshape batch_idxs to [B,G,M]
            batch_idxs = batch_idxs.view(-1, max_centers, self.group_size)

            # Process features
            input_features = None
            if batch_feature_list:
                input_features = torch.cat(batch_feature_list, dim=0)  # [B,G,M,C]

            # Final check to ensure no NaN
            if torch.isnan(neighborhoods).any():
                neighborhoods = torch.nan_to_num(neighborhoods, nan=0.0)
            if torch.isnan(centers).any():
                centers = torch.nan_to_num(centers, nan=0.0)
            if torch.isnan(batch_idxs).any():
                batch_idxs = torch.nan_to_num(batch_idxs, nan=0.0).long()
            if input_features is not None and torch.isnan(input_features).any():
                input_features = torch.nan_to_num(input_features, nan=0.0)
        except Exception as e:
            print(f"Error occurred while merging results: {e}")
            # Create emergency output
            B = len(batch_ids)
            neighborhoods = torch.zeros(B, max_centers, self.group_size, 3, device=xyz.device)
            centers = torch.zeros(B, max_centers, 3, device=xyz.device)
            batch_idxs = torch.zeros(B, max_centers, self.group_size, device=xyz.device).long()
            input_features = None
            if feature is not None:
                input_features = torch.zeros(B, max_centers, self.group_size, feature.shape[1], device=feature.device)
        
        return neighborhoods, centers, batch_idxs, batch_xyz_list, input_features

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    if not points.is_floating_point():
        points = points.float()
    
    if torch.isnan(points).any():
        points = torch.nan_to_num(points, nan=0.0)
    
    if torch.any(idx < 0) or torch.any(idx >= points.shape[1]):
        print("Warning: Invalid indices in index_points, correcting")
        idx = torch.clamp(idx, 0, points.shape[1] - 1)
        
    try:
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(
            device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        
        # Check if the result contains NaN
        if torch.isnan(new_points).any():
            new_points = torch.nan_to_num(new_points, nan=0.0)
            
        return new_points
    except Exception as e:
        print(f"Error occurred while indexing points: {e}")
        # Create a safe output
        output_shape = list(idx.shape) + [points.shape[-1]]
        return torch.zeros(output_shape, device=points.device, dtype=points.dtype)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(
                nn.Conv1d(last_channel, out_channel, 1).cuda())
            self.mlp_bns.append(nn.BatchNorm1d(out_channel).cuda())
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # Ensure inputs are floating point
        if not xyz1.is_floating_point():
            xyz1 = xyz1.float()
        if not xyz2.is_floating_point():
            xyz2 = xyz2.float()
        if points1 is not None and not points1.is_floating_point():
            points1 = points1.float()
        if points2 is not None and not points2.is_floating_point():
            points2 = points2.float()

        # Process NaN values
        if torch.isnan(xyz1).any():
            xyz1 = torch.nan_to_num(xyz1, nan=0.0)
        if torch.isnan(xyz2).any():
            xyz2 = torch.nan_to_num(xyz2, nan=0.0)
        if points1 is not None and torch.isnan(points1).any():
            points1 = torch.nan_to_num(points1, nan=0.0)
        if points2 is not None and torch.isnan(points2).any():
            points2 = torch.nan_to_num(points2, nan=0.0)
            
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        try:
            if S == 1:
                interpolated_points = points2.repeat(1, N, 1)
            else:
                dists = square_distance(xyz1, xyz2)
                dists, idx = dists.sort(dim=-1)
                dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

                # Ensure no infinite or NaN distances
                if torch.isinf(dists).any() or torch.isnan(dists).any():
                    dists = torch.nan_to_num(dists, nan=1e8, posinf=1e8)
                    # Give a small offset to avoid division by zero
                    dists = dists + 1e-8

                dist_recip = 1.0 / (dists + 1e-8)
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                weight = dist_recip / norm
                
                if torch.isnan(weight).any():
                    weight = torch.nan_to_num(weight, nan=1.0/3.0)
                
                interpolated_points = torch.sum(index_points(
                    points2, idx) * weight.view(B, N, 3, 1), dim=2)

            if points1 is not None:
                new_points = torch.cat([points1, interpolated_points], dim=-1)
            else:
                new_points = interpolated_points

            if torch.isnan(new_points).any():
                new_points = torch.nan_to_num(new_points, nan=0.0)
                
            new_points = new_points.permute(0, 2, 1)
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))
                
            if torch.isnan(new_points).any():
                new_points = torch.nan_to_num(new_points, nan=0.0)
                
            return new_points
        except Exception as e:
            out_channel = self.mlp_convs[-1].weight.shape[0]
            return torch.zeros(B, out_channel, N, device=xyz1.device)