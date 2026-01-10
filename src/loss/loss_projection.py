from dataclasses import dataclass
import numpy as np
import cv2

from jaxtyping import Float, Integer
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossProjectionCfg:
    weight: float


@dataclass
class LossProjectionCfgWrapper:
    projection: LossProjectionCfg


class LossProjection(Loss[LossProjectionCfg, LossProjectionCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        return_loss_map: bool = False,
        weights: Float[Tensor, "batch view"] | None = None,
    ):
        warp_by_bone = gaussians.lbs_weights_bones is not None
        b, v, _, h, w = batch["context"]["image"].shape
        means = gaussians.means
        lbs_weights = F.softmax(gaussians.lbs_weights, dim=-1)
        if warp_by_bone:
           lbs_weights_bones = F.softmax(gaussians.lbs_weights_bones, dim=-1)
        indices = gaussians.idx

        extrinsics = torch.linalg.inv(batch["context"]["extrinsics"])

        if return_loss_map:
            loss_map = torch.zeros_like(batch["context"]["image"][:, :, 0])

        loss = 0.
        for i in range(b):
            views_single = indices[i, ..., 0]

            means_single = means[i][views_single > 0]  # N x 3
            lbs_weights_single = lbs_weights[i][views_single > 0]  # N x J
            indices_single = indices[i][views_single > 0]  # N x 3

            indices_single = indices_single[:, [0, 2, 1]]  # change it view_id, w, h

            Rs_single = batch["context"]["Rs"][i][indices_single[:, 0] - 1]  # N x J x 3 x 3
            Ts_single = batch["context"]["Ts"][i][indices_single[:, 0] - 1]  # N x J x 3
            extrinsics_single = extrinsics[i][indices_single[:, 0] - 1]  # N x 4 x 4
            intrinsics_single = batch["context"]["intrinsics"][i][indices_single[:, 0] - 1]  # N x 3 x 3

            # apply lbs in context poses
            if warp_by_bone:
                lbs_weights_bones_single = lbs_weights_bones[i][views_single > 0]  # N x J

                cnl_Rs_single = batch["context"]["cnl_Rs"][i][indices_single[:, 0] - 1]  # N x J x 3 x 3
                cnl_Ts_single = batch["context"]["cnl_Ts"][i][indices_single[:, 0] - 1]  # N x J x 3

                means_single = (cnl_Rs_single @ means_single[:, None, :, None]).squeeze(-1) + cnl_Ts_single
                means_single = (means_single * lbs_weights_bones_single.unsqueeze(-1)).sum(1)
            means_pose = (Rs_single @ means_single[:, None, :, None]).squeeze(-1) + Ts_single  # N x J x 3
            means_pose = (means_pose * lbs_weights_single.unsqueeze(-1)).sum(1)  # N x 3

            # now project it to context views
            means_pose_cam = extrinsics_single[:, :3, :3] @ means_pose.unsqueeze(-1) + extrinsics_single[:, :3, 3:]
            intrinsics_single[..., 0, :] *= w
            intrinsics_single[..., 1, :] *= h
            means_pose_2d = (intrinsics_single @ means_pose_cam).squeeze(-1)
            means_pose_2d = means_pose_2d[..., :2] / means_pose_2d[..., 2:]
            means_pose_2d[:, 0] /= w
            means_pose_2d[:, 1] /= h

            indices_single = indices_single.float()
            indices_single[:, 1] /= w
            indices_single[:, 2] /= h
            
            # Compute per-point loss
            point_loss = (means_pose_2d - indices_single[:, 1:]) ** 2
            point_loss = point_loss.clip(max=1)  # [N, 2]
            point_loss = torch.mean(point_loss, dim=-1)  # [N] - mean over x,y coordinates
            
            # Apply per-view weights if provided
            if weights is not None:
                # indices_single[:, 0] contains view IDs (1-indexed), convert to 0-indexed
                view_ids = (indices_single[:, 0] - 1).long()  # [N] - view indices (0-indexed)
                # Get weights for each point based on its view
                point_weights = weights[i][view_ids]  # [N] - weight for each point
                # Weight the loss: multiply each point's loss by its view weight
                weighted_loss = point_loss * point_weights  # [N]
                loss += torch.mean(weighted_loss)  # Average weighted loss
            else:
                # No weights: use uniform weighting (original behavior)
                loss += torch.mean(point_loss)

            if return_loss_map:
                inds = indices[i][views_single > 0]
                inds = (inds[:, 0] - 1) * h * w + inds[:, 1] * w + inds[:, 2]
                loss_map_single = loss_map[i].reshape(-1)
                # Use the same point_loss computation for consistency
                if weights is not None:
                    # Apply weights to loss map as well
                    view_ids = (indices_single[:, 0] - 1).long()
                    point_weights = weights[i][view_ids]
                    loss_map_single[inds] = point_loss * point_weights
                else:
                    loss_map_single[inds] = point_loss
                loss_map[i] = loss_map_single.reshape(v, h, w)
        loss /= b

        if return_loss_map:
            return self.cfg.weight * loss, loss_map
        else:
            return self.cfg.weight * loss
