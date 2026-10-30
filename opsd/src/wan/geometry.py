"""Shared tensor-only camera geometry; backends select the coordinate policy.

WorldKV ray geometry is licensed under Apache-2.0. PRoPE routines below
retain their MIT license.
"""

from dataclasses import dataclass, replace
from functools import partial

import torch
from einops import rearrange


def relative_viewmats(viewmats):
    """Change the world frame to the first camera without rescaling translation.

    Input/output are world-to-camera matrices. Preserve the original FP32
    inverse/product ordering even inside a BF16 model autocast region.
    """
    with torch.autocast(device_type=viewmats.device.type, enabled=False):
        camera_to_world = torch.linalg.inv(viewmats.float())
        camera_to_world = torch.linalg.inv(camera_to_world[:1]) @ camera_to_world
        return torch.linalg.inv(camera_to_world).to(viewmats.dtype)


def camera_trajectory(sample, target_frames, device, dtype, *, reference, block_frames):
    """Prepare calibrated W2C/K tensors on the complete condition/target timeline."""
    prefix = len(sample['condition_viewmats'])
    for name in ('viewmats', 'intrinsics'):
        if len(sample[name]) < target_frames or len(sample[f'condition_{name}']) != prefix:
            raise ValueError('Camera inputs do not cover the prefix and requested target')
    views = torch.cat((sample['condition_viewmats'], sample['viewmats'][:target_frames]))
    views = views.to(device=device, dtype=torch.float32)
    with torch.autocast(device_type=views.device.type, enabled=False):
        if reference == 'initial':
            views = relative_viewmats(views)
        else:
            c2w = torch.linalg.inv(views)
            chunks = []
            for start in range(0, len(views), block_frames):
                anchor = max(start - 1, 0) if reference == 'previous_block' else 0
                poses = views[anchor] @ c2w[start:start + block_frames]
                if start == 0:
                    poses[0] = torch.eye(4, device=device)
                chunks.append(_invert_SE3(poses).to(dtype))
            views = torch.cat(chunks)
    intrinsics = torch.cat((sample['condition_intrinsics'], sample['intrinsics'][:target_frames]))
    return views.to(dtype)[None], intrinsics.to(device=device, dtype=dtype)[None]


def camera_view_segment(sample, start, length, device, dtype, *, reference, block_frames):
    """Prepare an online camera segment using the same trajectory reference."""
    prefix = len(sample['condition_viewmats'])
    views = sample['viewmats'][start-prefix:start-prefix+length].to(device=device, dtype=torch.float32)
    with torch.autocast(device_type=views.device.type, enabled=False):
        if reference == 'initial':
            initial = sample['condition_viewmats'][:1].to(device=device, dtype=torch.float32)
            views = relative_viewmats(torch.cat((initial, views)))[1:]
        else:
            chunks = []
            position = start
            while position < start + length:
                block_start = position // block_frames * block_frames
                end = min(block_start + block_frames, start + length)
                anchor = max(block_start - 1, 0) if reference == 'previous_block' else 0
                anchor_view = (sample['condition_viewmats'][anchor] if anchor < prefix else
                               sample['viewmats'][anchor-prefix]).to(device=device, dtype=torch.float32)
                poses = anchor_view @ torch.linalg.inv(views[position-start:end-start])
                chunks.append(_invert_SE3(poses).to(dtype))
                position = end
            views = torch.cat(chunks)
    return views.to(dtype)[None]


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    Rot = T[:, :3, :3]  # [B,3,3]
    trans = T[:, :3, 3:]  # [B,3,1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(
        T.shape[0], 1, 1
    )
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    # ensure identity matrix for 1st frame
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        # compute pose between i and i+1
        relative_poses_framewise = torch.bmm(
            SE3_inverse(relative_poses[:-1]), relative_poses[1:]
        )
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:  # Scale translations by their maximum displacement.
        translations = relative_poses[:, :3, 3]  # [f, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        # Normalize only when the camera moves.
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    bias: float = 0.5,
    device="cuda",
    dtype=torch.float32,
) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias  # [h*w, 2]
    grid_xy = grid_xy[None, ...].expand(n_frames, -1, -1)  # read-only frame views
    return grid_xy


def camera_ray_map(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
    *,
    representation: str = "origin_direction",
):
    """Build [F,H,W,C] rays from c2w and pixel-space [fx,fy,cx,cy].

    Origins/directions and Pluecker moment/directions are different checkpoint
    contracts. The backend selects one explicitly; no model names are inspected.
    """
    if representation not in ("origin_direction", "plucker", "direction"):
        raise ValueError(f"Unknown ray representation: {representation}")
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(
        n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype
    )  # [f, h*w, 2]
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)  # [f, 1]

    i = grid_xy[..., 0]  # [f, h*w]
    j = grid_xy[..., 1]  # [f, h*w]
    zs = torch.ones_like(i)  # [f, h*w]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)  # [f, h*w, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True)  # [f, h*w, 3]

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)  # [f, h*w, 3]
    if representation == "direction":
        features = rays_d
    else:
        rays_o = c2ws_mat[:, :3, 3]  # [f, 3]
        rays_o = rays_o[:, None, :].expand_as(rays_d)  # [f, h*w, 3]
        first = torch.cross(rays_o, rays_d, dim=-1) if representation == "plucker" else rays_o
        features = torch.cat([first, rays_d], dim=-1)
    return features.view(n_frames, height, width, features.shape[-1])


def pack_ray_map(rays, spatial_stride):
    """Pixel-unshuffle [F,H,W,C] into [1,C*s*s,F,H/s,W/s]."""
    return rearrange(
        rays, "f (h p) (w q) c -> 1 (c p q) f h w",
        p=spatial_stride, q=spatial_stride,
    )


@dataclass(frozen=True)
class CameraRays:
    """Prepared [B,F] c2w/pixel intrinsics; build dense rays only for a slice.

    The backend performs relative-pose conversion and translation normalization
    before constructing this object. Slicing never renormalizes a trajectory.
    Geometry retains its input precision independently of the model's autocast.
    """

    poses: torch.Tensor  # [B,F,4,4]
    intrinsics: torch.Tensor  # [B,F,4]: fx, fy, cx, cy in pixels
    height: int
    width: int
    representation: str = 'origin_direction'

    def __post_init__(self):
        if self.poses.ndim != 4 or self.poses.shape[-2:] != (4, 4):
            raise ValueError('Camera ray poses must have shape [B,F,4,4]')
        if self.intrinsics.shape != (*self.poses.shape[:2], 4):
            raise ValueError('Camera ray intrinsics must have shape [B,F,4]')
        if self.poses.device != self.intrinsics.device:
            raise ValueError('Camera poses and intrinsics must be on the same device')
        if any(type(size) is not int or size <= 0 for size in (self.height, self.width)):
            raise ValueError('Camera ray image dimensions must be positive integers')

    def __getitem__(self, index):
        return replace(self, poses=self.poses[index], intrinsics=self.intrinsics[index])

    def packed(self, spatial_stride, *, dtype=None):
        """Materialize [B,C*s*s,F,H/s,W/s] in native ray/channel order."""
        if type(spatial_stride) is not int or spatial_stride <= 0:
            raise ValueError('Ray packing stride must be a positive integer')
        if self.height % spatial_stride or self.width % spatial_stride:
            raise ValueError('Camera ray image dimensions must align with the packing stride')
        with torch.autocast(device_type=self.poses.device.type, enabled=False):
            rays = camera_ray_map(
                self.poses.flatten(0, 1), self.intrinsics.flatten(0, 1), self.height, self.width,
                representation=self.representation,
            )
            packed = rearrange(pack_ray_map(rays, spatial_stride), '1 c (b f) h w -> b c f h w',
                               b=self.poses.shape[0])
            return packed if dtype is None else packed.to(dtype=dtype)


# MIT License
#
# Copyright (c) Authors of
# "PRoPE: Projective Positional Encoding for Multiview Transformers"
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def projective_qkv(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: torch.Tensor,  # (batch, cameras, 3, 3)
):
    """Transform Q/K/V using camera projection matrices; return output correction."""
    # We're going to assume self-attention: all inputs are the same shape.
    batch, _, _, head_dim = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks.shape == (batch, cameras, 3, 3)

    P, P_inv = projective_matrices(viewmats, Ks)
    P_T = P.transpose(-1, -2).to(dtype=viewmats.dtype)
    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)
    assert head_dim % 4 == 0

    def apply_matrix(feats, matrix):
        assert feats.shape[-1] == head_dim
        # Native single-segment cat fixes output strides; contiguous is not equivalent.
        return torch.cat([_apply_tiled_projmat(feats, matrix)], dim=-1)

    return (apply_matrix(q, P_T), apply_matrix(k, P_inv), apply_matrix(v, P_inv),
            partial(apply_matrix, matrix=P))


def projective_matrices(viewmats, Ks):
    """Build PRoPE projection matrices and their inverses."""

    # Normalize camera intrinsics.
    Ks_norm = torch.zeros_like(Ks)
    Ks_norm[..., 0, 0] = Ks[..., 0, 0]
    Ks_norm[..., 1, 1] = Ks[..., 1, 1]
    Ks_norm[..., 0, 2] = 0
    Ks_norm[..., 1, 2] = 0
    Ks_norm[..., 2, 2] = 1.0
    Ks_norm = Ks_norm.to(dtype=Ks.dtype)

    # P maps world coordinates to homogeneous image coordinates.
    P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
    P_inv = torch.einsum(
        "...ij,...jk->...ik",
        _invert_SE3(viewmats),
        _lift_K(_invert_K(Ks_norm)),
    ).to(dtype=viewmats.dtype)

    return P, P_inv


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen >= cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    out = out.to(dtype=transforms.dtype)
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    out = out.to(dtype=Ks.dtype)
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    out = out.to(dtype=Ks.dtype)
    return out
