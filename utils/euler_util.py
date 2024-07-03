import torch
from pytorch3d.ops import box3d_overlap
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss
from typing import Union, Tuple
import time
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles

def bbox_to_corners(centers, sizes, rot_mat: torch.Tensor) -> torch.Tensor:
    """Transform bbox parameters to the 8 corners.

    Args:
        bbox (Tensor): 3D box of shape (N, 6) or (N, 7) or (N, 9).

    Returns:
        Tensor: Transformed 3D box of shape (N, 8, 3).
    """
    device = centers.device
    use_batch = False
    if len(centers.shape) == 3:
        use_batch = True
        batch_size, n_proposals = centers.shape[0], centers.shape[1]
        centers = centers.reshape(-1, 3)
        sizes = sizes.reshape(-1, 3)
        rot_mat = rot_mat.reshape(-1, 3, 3)
        
    n_box = centers.shape[0]
    if use_batch:
        assert n_box == batch_size * n_proposals
    centers = centers.unsqueeze(1).repeat(1, 8, 1)  # shape (N, 8, 3)
    half_sizes = sizes.unsqueeze(1).repeat(1, 8, 1) / 2  # shape (N, 8, 3)
    eight_corners_x = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1],
                                   device=device).unsqueeze(0).repeat(n_box, 1)  # shape (N, 8)
    eight_corners_y = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1],
                                   device=device).unsqueeze(0).repeat(
                                       n_box, 1)  # shape (N, 8)
    eight_corners_z = torch.tensor([1, -1, -1, 1, 1, -1, -1, 1],
                                   device=device).unsqueeze(0).repeat(
                                       n_box, 1)  # shape (N, 8)
    eight_corners = torch.stack(
        (eight_corners_x, eight_corners_y, eight_corners_z),
        dim=-1)  # shape (N, 8, 3)
    eight_corners = eight_corners * half_sizes  # shape (N, 8, 3)
    # rot_mat: (N, 3, 3), eight_corners: (N, 8, 3)
    rotated_corners = torch.matmul(eight_corners,
                                   rot_mat.transpose(1, 2))  # shape (N, 8, 3)
    res = centers + rotated_corners
    if use_batch:
        res = res.reshape(batch_size, n_proposals, 8, 3)
    return res

def euler_iou3d(boxes1, boxes2):
    rows = boxes1.shape[0]
    cols = boxes2.shape[0]
    if rows * cols == 0:
        return boxes1.tensor.new(rows, cols)
    
    _, iou3d = box3d_overlap(boxes1, boxes2)
    return iou3d

def euler_iou3d_split(center1, size1, rot1, center2, size2, rot2):
    device = center1.device
    center1 = center1.cuda()
    size1 = size1.cuda()
    rot1 = rot1.cuda()
    center2 = center2.cuda()
    size2 = size2.cuda()
    rot2 = rot2.cuda()
    corners1 = bbox_to_corners(center1, size1, rot1)
    corners2 = bbox_to_corners(center2, size2, rot2)
    return euler_iou3d(corners1, corners2).to(device)

def chamfer_distance(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_weight: Union[torch.Tensor, float] = 1.0,
        dst_weight: Union[torch.Tensor, float] = 1.0,
        criterion_mode: str = 'l2',
        reduction: str = 'mean') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate Chamfer Distance of two sets.

    Args:
        src (Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (Tensor or float): Weight of source loss. Defaults to 1.0.
        dst_weight (Tensor or float): Weight of destination loss.
            Defaults to 1.0.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are 'smooth_l1', 'l1' or 'l2'. Defaults to 'l2'.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.
            Defaults to 'mean'.

    Returns:
        tuple: Source and Destination loss with the corresponding indices.

            - loss_src (Tensor): The min distance
              from source to destination.
            - loss_dst (Tensor): The min distance
              from destination to source.
            - indices1 (Tensor): Index the min distance point
              for each point in source to destination.
            - indices2 (Tensor): Index the min distance point
              for each point in destination to source.
    """
    if len(src.shape) == 4:
        src = src.reshape(-1, 8, 3)
    if len(dst.shape) == 4:
        dst = dst.reshape(-1, 8, 3)

    if criterion_mode == 'smooth_l1':
        criterion = smooth_l1_loss
    elif criterion_mode == 'l1':
        criterion = l1_loss
    elif criterion_mode == 'l2':
        criterion = mse_loss
    else:
        raise NotImplementedError

    src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)

    distance = criterion(src_expand, dst_expand, reduction='none').sum(-1)
    src2dst_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
    dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)

    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)

    if reduction == 'sum':
        loss_src = torch.sum(loss_src)
        loss_dst = torch.sum(loss_dst)
    elif reduction == 'mean':
        loss_src = torch.mean(loss_src)
        loss_dst = torch.mean(loss_dst)
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError

    return loss_src, loss_dst, indices1, indices2

def euler_to_matrix_np(euler):
    # euler: N*3 np array
    euler_tensor = torch.tensor(euler)
    matrix_tensor = euler_angles_to_matrix(euler_tensor, 'ZXY')
    return matrix_tensor.numpy()

def matrix_to_euler_np(matrix):
    matrix_tensor = torch.tensor(matrix)
    euler_tensor = matrix_to_euler_angles(matrix_tensor, 'ZXY')
    return euler_tensor.numpy()