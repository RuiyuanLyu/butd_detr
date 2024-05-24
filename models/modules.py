# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2.pointnet2_utils import gather_operation


class PointsObjClsModule(nn.Module):

    def __init__(self, seed_feature_dim):
        """
        Object candidate point prediction from seed point features.

        Args:
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.conv2 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn2 = nn.BatchNorm1d(self.in_dim)
        self.conv3 = nn.Conv1d(self.in_dim, 1, 1)

    def forward(self, seed_features):
        """ Forward pass.

        Arguments:
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            logits: (batch_size, 1, num_seed)
        """
        net = F.relu(self.bn1(self.conv1(seed_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        logits = self.conv3(net)  # (batch_size, 1, num_seed)

        return logits


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class GeneralSamplingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz, features, sample_inds):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        """
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = gather_operation(
            xyz_flipped, sample_inds
        ).transpose(1, 2).contiguous()
        new_features = gather_operation(features, sample_inds).contiguous()

        return new_xyz, new_features, sample_inds


class ThreeLayerMLP(nn.Module):
    """A 3-layer MLP with normalization and dropout."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x):
        """Forward pass, x can be (B, dim, N)."""
        return self.net(x)


class ClsAgnosticPredictHead(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_proposal,
                 seed_feat_dim=256, objectness=True, heading=0,
                 compute_sem_scores=True):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim
        self.objectness = objectness
        self.heading = heading
        self.compute_sem_scores = compute_sem_scores

        if objectness:
            self.objectness_scores_head = ThreeLayerMLP(seed_feat_dim, 1)
        self.center_residual_head = ThreeLayerMLP(seed_feat_dim, 3)
        if heading: 
            self.rotmat_pred_head = ThreeLayerMLP(seed_feat_dim, 6)
        self.size_pred_head = ThreeLayerMLP(seed_feat_dim, 3)
        if compute_sem_scores:
            self.sem_cls_scores_head = ThreeLayerMLP(seed_feat_dim, self.num_class)

    def forward(self, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,C,num_proposal)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[-1]
        net = features

        # objectness
        if self.objectness:
            objectness_scores = self.objectness_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 1)
            end_points[f'{prefix}objectness_scores'] = objectness_scores.squeeze(-1)

        # center
        center_residual = self.center_residual_head(net).transpose(2, 1)
        center = base_xyz + center_residual  # (batch_size, num_proposal, 3)

        # heading
        if self.heading:
            rotmat_pred = self.rotmat_pred_head(net).transpose(2, 1).view(
                [batch_size, num_proposal, 6])  # (batch_size, num_proposal, 6)
            x_raw, y_raw = rotmat_pred[..., :3], rotmat_pred[..., 3:] # (batch_size, num_proposal, 3)
            rot_mat = ortho_6d_2_Mat(x_raw.view(batch_size, num_proposal, 3), y_raw.view(batch_size, num_proposal, 3))
            euler = matrix_to_euler_angles(rot_mat, 'ZXY')
            end_points[f'{prefix}rot_mat'] = rot_mat
            end_points[f'{prefix}euler'] = euler

        # size
        pred_size = self.size_pred_head(net).transpose(2, 1).view(
            [batch_size, num_proposal, 3])  # (batch_size, num_proposal, 3)

        # class
        if self.compute_sem_scores:
            sem_cls_scores = self.sem_cls_scores_head(features).transpose(2, 1)  # (batch_size, num_proposal, num_class)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}pred_size'] = pred_size

        if self.compute_sem_scores:
            end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores
        return center, pred_size


def normalize_vector(vector):
    norm = torch.norm(vector, dim=1, keepdim=True) + 1e-8
    normalized_vector = vector / norm
    return normalized_vector


def cross_product(a, b):
    cross_product = torch.cross(a, b, dim=1)
    return cross_product


def ortho_6d_2_Mat(x_raw, y_raw):
    """x_raw, y_raw: both tensors (batch, 3)."""
    y = normalize_vector(y_raw)
    z = cross_product(x_raw, y)
    z = normalize_vector(z)  # (batch, 3)
    x = cross_product(y, z)  # (batch, 3)

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    matrix = torch.cat((x, y, z), 2)  # (batch, 3)
    return matrix

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)
