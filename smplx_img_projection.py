"""
Project an SMPL-X mesh / joints onto an image for camera sanity-checking.

Key conventions:
- This repo's training pipeline commonly uses:
  - intrinsics as *normalized* K (fx, fy, cx, cy normalized by W/H)
  - extrinsics as c2w (camera-to-world) in several datasets (e.g. THuman returns w2c.inverse()).
- For 2D projection, we need a *pixel* K and a w2c [R|t] mapping world -> camera.

This script supports both:
- extrinsics_type: c2w or w2c
- intrinsics_type: normalized or pixel
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Literal

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from smplx import SMPLX

from utils.easymocap2smplx import convert_easymocap_to_smplx

class SMPLXProjectionVisualizer:
    """
    Visualize SMPLX model projected onto images using camera parameters.
    
    Expected data format:
    - Camera intrinsics: 3x3 matrix or dict with fx, fy, cx, cy
    - Camera extrinsics: 4x4 transformation matrix (world to camera) or R, t separately
    - SMPLX params: dict with keys like 'betas', 'body_pose', 'global_orient', etc.
    """
    
    def __init__(
        self,
        smplx_model_path: str,
        gender: Literal["male", "female", "neutral"] = "neutral",
        use_pca: bool = False,
        flat_hand_mean: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the visualizer.
        
        Args:
            smplx_model_path: Path to SMPLX model directory
            gender: 'male', 'female', or 'neutral'
        """
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # In this repo, SMPLX is typically loaded from the *.npz model file directly.
        # Also, your EasyMocap->SMPLX conversion produces full 45-dim hand poses,
        # so we disable PCA by default (use_pca=False), matching the model wrapper.
        self.smplx_model = SMPLX(
            model_path=smplx_model_path,
            gender=gender,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
        ).to(self.device)
        
    def parse_camera_params(
        self,
        intrinsics: Any,
        extrinsics: Any,
        image_wh: Optional[Tuple[int, int]] = None,
        intrinsics_type: Literal["pixel", "normalized"] = "pixel",
        extrinsics_type: Literal["w2c", "c2w"] = "c2w",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse camera parameters into standard format.
        
        Args:
            intrinsics: 3x3 matrix or dict with fx, fy, cx, cy
            extrinsics: 4x4 matrix or dict with 'R' and 't'
            
        Returns:
            K (3x3), R (3x3), t (3x1)
        """
        # ---- Intrinsics ----
        # Accept dict or 3x3. If normalized, requires image_wh to convert to pixels.
        if isinstance(intrinsics, dict):
            fx = intrinsics.get("fx", intrinsics.get("focal_length_x"))
            fy = intrinsics.get("fy", intrinsics.get("focal_length_y"))
            cx = intrinsics.get("cx", intrinsics.get("principal_point_x"))
            cy = intrinsics.get("cy", intrinsics.get("principal_point_y"))
            if fx is None or fy is None or cx is None or cy is None:
                raise ValueError(f"Intrinsics dict must contain fx,fy,cx,cy (or focal_length_x/y, principal_point_x/y). Got keys={list(intrinsics.keys())}")
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        else:
            K = np.array(intrinsics, dtype=np.float32)
            if K.shape != (3, 3):
                raise ValueError(f"Intrinsics must be dict or 3x3 matrix. Got shape={K.shape}")

        if intrinsics_type == "normalized":
            if image_wh is None:
                raise ValueError("intrinsics_type='normalized' requires image_wh=(W,H) to convert to pixel K.")
            W, H = image_wh
            K_px = K.copy()
            K_px[0, :] *= W
            K_px[1, :] *= H
            K = K_px
            
        # ---- Extrinsics ----
        # We need world->camera (w2c) as [R|t] for projection.
        if isinstance(extrinsics, dict):
            # Supports either {"R":..., "t":...} or {"rotation":..., "translation":...}
            if "R" in extrinsics and "t" in extrinsics:
                E = np.eye(4, dtype=np.float32)
                E[:3, :3] = np.array(extrinsics["R"], dtype=np.float32)
                t_vec = np.array(extrinsics["t"], dtype=np.float32).reshape(3)
                E[:3, 3] = t_vec
            elif "rotation" in extrinsics and "translation" in extrinsics:
                E = np.eye(4, dtype=np.float32)
                E[:3, :3] = np.array(extrinsics["rotation"], dtype=np.float32)
                t_vec = np.array(extrinsics["translation"], dtype=np.float32).reshape(3)
                E[:3, 3] = t_vec
            else:
                raise ValueError(f"Extrinsics dict must contain (R,t) or (rotation,translation). Got keys={list(extrinsics.keys())}")
        else:
            E = np.array(extrinsics, dtype=np.float32)
            if E.shape != (4, 4):
                raise ValueError(f"Extrinsics must be dict or 4x4 matrix. Got shape={E.shape}")

        if extrinsics_type == "c2w":
            E = np.linalg.inv(E)  # convert to w2c

        R = E[:3, :3]
        t = E[:3, 3:4]
                
        return K, R, t
    
    def get_smplx_output(self, smplx_params: Dict) -> Dict[str, np.ndarray]:
        """
        Get SMPLX vertices from parameters.
        
        Args:
            smplx_params: Dictionary with SMPLX parameters
            
        Returns:
            vertices: (N, 3) array of vertex positions
        """
        # Prepare parameters
        params = {}
        
        # Handle different parameter formats
        for key in ['betas', 'body_pose', 'global_orient', 'transl', 
                    'left_hand_pose', 'right_hand_pose', 'jaw_pose', 
                    'leye_pose', 'reye_pose', 'expression']:
            if key in smplx_params:
                val = smplx_params[key]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32)
                params[key] = val.to(self.device)
        
        # Ensure proper shapes
        if 'betas' in params and params['betas'].dim() == 1:
            params['betas'] = params['betas'].unsqueeze(0)
        if 'body_pose' in params and params['body_pose'].dim() == 1:
            params['body_pose'] = params['body_pose'].unsqueeze(0)
        if 'global_orient' in params and params['global_orient'].dim() == 1:
            params['global_orient'] = params['global_orient'].unsqueeze(0)
            
        # Get SMPLX output
        with torch.no_grad():
            output = self.smplx_model(**params)
        vertices = output.vertices.detach().cpu().numpy()[0]
        joints = output.joints.detach().cpu().numpy()[0]
        return {"vertices": vertices, "joints": joints}
    
    def project_points(self, points_world: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Project 3D vertices to 2D image coordinates.
        
        Args:
            vertices: (N, 3) world coordinates
            K: (3, 3) intrinsic matrix
            R: (3, 3) rotation matrix
            t: (3, 1) translation vector
            
        Returns:
            projected: (N, 2) image coordinates
        """
        points_cam = (R @ points_world.T + t).T  # (N,3)
        points_img = (K @ points_cam.T).T
        points_img = points_img[:, :2] / points_img[:, 2:3]
        return points_img
    
    def visualize(
        self,
        image_path: str,
        intrinsics: Any,
        extrinsics: Any,
        smplx_params: Dict,
        output_path: Optional[str] = None,
        point_size: int = 2,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.7,
        show: bool = True,
        draw: Literal["joints", "vertices"] = "joints",
        max_points: int = 20000,
        intrinsics_type: Literal["pixel", "normalized"] = "pixel",
        extrinsics_type: Literal["w2c", "c2w"] = "w2c",
    ):
        """
        Visualize SMPLX projected onto an image.
        
        Args:
            image_path: Path to input image
            intrinsics: Camera intrinsics
            extrinsics: Camera extrinsics
            smplx_params: SMPLX parameters
            output_path: Path to save visualization (optional)
            point_size: Size of projected points
            color: RGB color for visualization
            show_wireframe: Whether to draw mesh edges
            alpha: Transparency for overlay
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]

        # Parse camera parameters (convert normalized intrinsics using image size)
        K, R, t = self.parse_camera_params(
            intrinsics,
            extrinsics,
            image_wh=(w, h),
            intrinsics_type=intrinsics_type,
            extrinsics_type=extrinsics_type,
        )

        # Get SMPLX output
        smplx_out = self.get_smplx_output(smplx_params)
        points = smplx_out["joints"] if draw == "joints" else smplx_out["vertices"]
        if draw == "vertices" and points.shape[0] > max_points:
            # Subsample for speed
            idx = np.random.choice(points.shape[0], size=max_points, replace=False)
            points = points[idx]

        points_2d = self.project_points(points, K, R, t)
        
        # Create visualization
        overlay = img.copy()
        
        # Draw points
        for p2d in points_2d:
            x, y = int(p2d[0]), int(p2d[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(overlay, (x, y), point_size, color, -1)
        
        # Blend with original image
        result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        
        fig = plt.figure(figsize=(12, 8))
        plt.imshow(result)
        plt.title("SMPLX Projection Visualization")
        plt.axis("off")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return result, fig


def load_smplx_params(path: str, easymocap_default_model: str, zero_hands: bool = False) -> Dict[str, np.ndarray]:
    """
    Load SMPL-X parameters from:
    - *.npz (keys betas/body_pose/global_orient/transl/...)
    - *.pkl/*.pickle (dict)
    - *.json:
        - EasyMocap json (list or dict) -> converted via convert_easymocap_to_smplx
        - already-converted SMPLX dict json
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        return {k: data[k] for k in data.files}
    if p.suffix in [".pkl", ".pickle"]:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Pickle must contain a dict, got {type(obj)}")
        return obj
    if p.suffix == ".json":
        with open(p, "r") as f:
            obj = json.load(f)
        # EasyMocap often stores as [ {...} ]
        if isinstance(obj, list):
            obj0 = obj[0]
        else:
            obj0 = obj
        # If it's EasyMocap-style, it has 'poses'/'Rh'/'Th'
        if isinstance(obj0, dict) and ("poses" in obj0 and "Rh" in obj0 and "Th" in obj0):
            return convert_easymocap_to_smplx(obj0, easymocap_default_model, zero_hands=zero_hands)
        if not isinstance(obj0, dict):
            raise ValueError(f"JSON must contain a dict or list[dict], got {type(obj0)}")
        return obj0

    raise ValueError(f"Unsupported SMPLX params file extension: {p.suffix}")


def load_matrix(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix == ".npy":
        arr = np.load(p)
        return arr
    if p.suffix == ".json":
        with open(p, "r") as f:
            obj = json.load(f)
        return np.array(obj, dtype=np.float32)
    raise ValueError("Matrix must be .npy or .json (3x3 or 4x4 list).")


def plot_smplx_projection_notebook(
    image_path: str,
    smplx_model_path: str,
    smplx_params: Dict[str, Any],
    intrinsics: Any,
    extrinsics: Any,
    intrinsics_type: Literal["pixel", "normalized"] = "pixel",
    extrinsics_type: Literal["w2c", "c2w"] = "w2c",
    draw: Literal["joints", "vertices"] = "joints",
    point_size: int = 2,
    alpha: float = 0.7,
    color: Tuple[int, int, int] = (0, 255, 0),
    show: bool = True,
):
    """
    Notebook-friendly helper. Returns (rgb_image, matplotlib_figure).
    """
    viz = SMPLXProjectionVisualizer(smplx_model_path, gender="neutral", use_pca=False, flat_hand_mean=True)
    result, fig = viz.visualize(
        image_path=image_path,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        smplx_params=smplx_params,
        output_path=None,
        point_size=point_size,
        color=color,
        alpha=alpha,
        show=show,
        draw=draw,
        intrinsics_type=intrinsics_type,
        extrinsics_type=extrinsics_type,
    )
    return result, fig


def main():
    parser = argparse.ArgumentParser(description="Project SMPL-X onto an image for camera sanity checking.")
    parser.add_argument("--image", required=True, type=str, help="Path to RGB image.")
    parser.add_argument("--smplx-model", required=True, type=str, help="Path to SMPL-X model *.npz (e.g. datasets/smplx/SMPLX_MALE.npz).")

    parser.add_argument("--smplx-params", required=True, type=str, help="Path to SMPL-X params (.json EasyMocap or SMPLX dict, .npz, or .pkl).")
    parser.add_argument("--easymocap-default-model", type=str, default="datasets/smplx/SMPLX_MALE.npz", help="SMPL-X model for EasyMocap conversion.")
    parser.add_argument("--zero-hands", action="store_true", help="Zero hand pose when converting EasyMocap -> SMPLX.")

    # Camera inputs
    parser.add_argument("--intrinsics", required=True, type=str, help="Intrinsics: path to .npy/.json (3x3) OR a JSON string dict with fx/fy/cx/cy.")
    parser.add_argument("--extrinsics", required=True, type=str, help="Extrinsics: path to .npy/.json (4x4) OR a JSON string dict with R/t (or rotation/translation).")
    parser.add_argument("--intrinsics-type", choices=["pixel", "normalized"], default="pixel", help="Whether provided intrinsics are pixel K or normalized K.")
    parser.add_argument("--extrinsics-type", choices=["w2c", "c2w"], default="w2c", help="Whether provided extrinsics are world->cam or cam->world.")

    parser.add_argument("--draw", choices=["joints", "vertices"], default="joints")
    parser.add_argument("--point-size", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--output", type=str, default=None, help="Optional output image path (png).")
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    # Load camera params. Accept file path or inline JSON.
    def parse_maybe_inline_json(s: str):
        s = s.strip()
        if s.startswith("{") or s.startswith("["):
            return json.loads(s)
        mat = load_matrix(s)
        if mat is not None:
            return mat
        raise ValueError(f"Could not parse camera param: {s}")

    intr = parse_maybe_inline_json(args.intrinsics)
    extr = parse_maybe_inline_json(args.extrinsics)

    smplx_params = load_smplx_params(args.smplx_params, args.easymocap_default_model, zero_hands=args.zero_hands)

    viz = SMPLXProjectionVisualizer(args.smplx_model, gender="neutral", use_pca=False, flat_hand_mean=True)
    viz.visualize(
        image_path=args.image,
        intrinsics=intr,
        extrinsics=extr,
        smplx_params=smplx_params,
        output_path=args.output,
        point_size=args.point_size,
        alpha=args.alpha,
        show=not args.no_show,
        draw=args.draw,
        intrinsics_type=args.intrinsics_type,
        extrinsics_type=args.extrinsics_type,
    )


if __name__ == "__main__":
    main()


# Example usage
if __name__ == "__main__":
    # Initialize visualizer
    smplx_model_path = "/path/to/smplx/models"  # Update this path
    visualizer = SMPLXProjectionVisualizer(smplx_model_path, gender='neutral')
    
    # Example camera parameters (update with your actual values)
    intrinsics = {
        'fx': 1000.0,
        'fy': 1000.0,
        'cx': 512.0,
        'cy': 512.0
    }
    
    # Or as a matrix:
    # intrinsics = np.array([[1000, 0, 512],
    #                        [0, 1000, 512],
    #                        [0, 0, 1]])
    
    extrinsics = {
        'R': np.eye(3),  # Identity rotation
        't': np.array([0, 0, 5])  # 5 meters away from camera
    }
    
    # Or as 4x4 matrix:
    # extrinsics = np.eye(4)
    # extrinsics[:3, 3] = [0, 0, 5]
    
    # Example SMPLX parameters (update with your actual values)
    smplx_params = {
        'betas': np.zeros(10),
        'body_pose': np.zeros(63),
        'global_orient': np.zeros(3),
        'transl': np.array([0, 0, 0])
    }
    
    # Or load from file:
    # with open('smplx_params.pkl', 'rb') as f:
    #     smplx_params = pickle.load(f)
    
    # Visualize
    image_path = "path/to/your/image.jpg"
    visualizer.visualize(
        image_path=image_path,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        smplx_params=smplx_params,
        output_path="projection_visualization.png",
        show_wireframe=True,
        alpha=0.5
    )