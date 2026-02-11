import json
import subprocess
import sys
import os
import pickle
import signal
import gc
from pathlib import Path
from typing import Literal, TypedDict, Optional
import argparse
import multiprocessing as mp
from multiprocessing import Queue, Lock, Value
import threading

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
import seaborn as sns
import trimesh

import cv2
from PIL import Image
from io import BytesIO

import nvdiffrast
import nvdiffrast.torch

from ..misc.body_utils import get_canonical_global_tfms, get_global_RTs, body_pose_to_body_RTs, apply_global_tfm_to_camera, apply_lbs_to_means

DEBUG = False
THuman21 = True
RASTERIZE_LBS_WEIGHTS = False

INPUT_DIR = Path("/workspace/humanvol/thuman") # assuming we already have this
if THuman21:
    OUTPUT_DIR = Path("/workspace/humanvol/thuman2.1")
else:
    OUTPUT_DIR = Path("/workspace/humanvol/thuman2.0")
FACE_CROP_JSON = Path("/workspace/humanvol/thuman/thuman_face_bboxes_val.json")

TRAIN_FRAME_ORDERS = []
for i in range(16):
    TRAIN_FRAME_ORDERS.append(i)
    TRAIN_FRAME_ORDERS.append(16 + i * 3)
    TRAIN_FRAME_ORDERS.append(16 + i * 3 + 1)
    TRAIN_FRAME_ORDERS.append(16 + i * 3 + 2)
TEST_FRAME_ORDERS = [0, 1, 2, 3, 4, 5]

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(5e7)


def get_example_keys(stage: Literal["test", "train"], custom_json_path: Optional[str] = None, limit: Optional[int] = None) -> list[str]:
    """Get example keys for a given stage."""
    if custom_json_path:
        with open(custom_json_path) as f:
            keys = json.load(f)
    elif THuman21 and stage == "train":
        with open(f'datasets/thuman/thuman2.1_train.json') as f:
            keys = json.load(f)
    else:
        with open(f'datasets/thuman/thuman2.0_{stage}.json') as f:
            keys = json.load(f)
    
    if limit is not None:
        keys = keys[:limit]
    
    return keys


def get_keys_from_torch_files(torch_files: list[str], stage_path: Path) -> set[str]:
    """
    Extract all keys from specified torch files using index.json.
    
    Args:
        torch_files: List of torch filenames (e.g., ["000332.torch", "000333.torch"])
        stage_path: Path to the stage directory containing the torch files and index.json
        
    Returns:
        Set of keys that were in those torch files
    """
    keys_to_reprocess = set()
    torch_file_set = set(torch_files)
    
    index_path = stage_path / "index.json"
    if not index_path.exists():
        print(f"Warning: index.json not found at {index_path}. Falling back to loading torch files...")
        # Fallback to old method if index.json doesn't exist
        for torch_file in tqdm(torch_files, desc="Loading torch files"):
            torch_path = stage_path / torch_file
            if not torch_path.exists():
                print(f"Warning: {torch_file} not found, skipping...")
                continue
                
            try:
                chunk = torch.load(torch_path)
                for example in chunk:
                    if "key" in example:
                        keys_to_reprocess.add(example["key"])
            except Exception as e:
                print(f"Warning: Could not load {torch_file}: {e}")
                continue
    else:
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)
            
            # Reverse lookup: find all keys that map to the specified torch files
            for key, torch_file in index.items():
                if torch_file in torch_file_set:
                    keys_to_reprocess.add(key)
            
            # Check if any torch files weren't found in the index
            found_torch_files = set(index.values()) & torch_file_set
            missing_torch_files = torch_file_set - found_torch_files
            if missing_torch_files:
                print(f"Warning: {len(missing_torch_files)} torch files not found in index.json: {sorted(list(missing_torch_files))[:5]}...")
        
        except Exception as e:
            print(f"Warning: Could not load index.json: {e}. Falling back to loading torch files...")
            # Fallback to old method if index.json can't be loaded
            for torch_file in tqdm(torch_files, desc="Loading torch files"):
                torch_path = stage_path / torch_file
                if not torch_path.exists():
                    print(f"Warning: {torch_file} not found, skipping...")
                    continue
                    
                try:
                    chunk = torch.load(torch_path)
                    for example in chunk:
                        if "key" in example:
                            keys_to_reprocess.add(example["key"])
                except Exception as e:
                    print(f"Warning: Could not load {torch_file}: {e}")
                    continue
    
    print(f"Found {len(keys_to_reprocess)} keys to reprocess from {len(torch_files)} torch files")
    return keys_to_reprocess


def delete_torch_files(torch_files: list[str], stage_path: Path, backup: bool = True):
    """
    Delete specified torch files, optionally backing them up first.
    
    Args:
        torch_files: List of torch filenames to delete
        stage_path: Path to the stage directory
        backup: If True, move files to a backup directory instead of deleting
    """
    if backup:
        backup_dir = stage_path / "backup_torch_files"
        backup_dir.mkdir(exist_ok=True, parents=True)
        print(f"Backing up {len(torch_files)} torch files to {backup_dir}...")
    else:
        print(f"Deleting {len(torch_files)} torch files...")
    
    for torch_file in torch_files:
        torch_path = stage_path / torch_file
        if not torch_path.exists():
            continue
            
        try:
            if backup:
                backup_path = backup_dir / torch_file
                torch_path.rename(backup_path)
            else:
                torch_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete/backup {torch_file}: {e}")


def get_already_processed_keys(stage: str, output_dir: Path, exclude_torch_files: Optional[list[str]] = None) -> set[str]:
    """
    Get set of keys that have already been processed.
    
    Args:
        stage: "train", "val", or "test"
        output_dir: Output directory path
        exclude_torch_files: List of torch files to exclude from "already processed" (for reprocessing)
        
    Returns:
        Set of already-processed keys
    """
    already_processed = set()
    stage_path = output_dir / stage
    
    if not stage_path.exists():
        return already_processed
    
    exclude_set = set(exclude_torch_files) if exclude_torch_files else set()
    
    index_path = stage_path / "index.json"
    if index_path.exists():
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)
                # Only include keys that are NOT in excluded torch files
                for key, torch_file in index.items():
                    if torch_file not in exclude_set:
                        already_processed.add(key)
                print(f"Found {len(already_processed)} already-processed keys from index.json (excluding {len(exclude_set)} torch files)")
                return already_processed
        except Exception as e:
            print(f"Warning: Could not load index.json: {e}. Scanning chunks instead...")
    
    # Otherwise, scan all chunk files (excluding specified ones)
    chunk_files = list(stage_path.glob("*.torch"))
    if not chunk_files:
        return already_processed
    
    print(f"Scanning {len(chunk_files)} existing chunk files to find already-processed keys...")
    for chunk_path in tqdm(chunk_files, desc="Loading chunks"):
        # Skip excluded torch files
        if chunk_path.name in exclude_set:
            continue
            
        try:
            chunk = torch.load(chunk_path)
            for example in chunk:
                if "key" in example:
                    already_processed.add(example["key"])
        except Exception as e:
            print(f"Warning: Could not load chunk {chunk_path}: {e}. Skipping...")
            continue
    
    print(f"Found {len(already_processed)} already-processed keys from chunks (excluding {len(exclude_set)} torch files)")
    return already_processed


def get_next_chunk_index(stage: str, output_dir: Path) -> int:
    """Get the next chunk index by finding the highest existing chunk number."""
    stage_path = output_dir / stage
    if not stage_path.exists():
        return 0
    
    chunk_files = list(stage_path.glob("*.torch"))
    if not chunk_files:
        return 0
    
    chunk_indices = []
    for chunk_path in chunk_files:
        try:
            chunk_idx = int(chunk_path.stem)
            chunk_indices.append(chunk_idx)
        except ValueError:
            continue
    
    if not chunk_indices:
        return 0
    
    return max(chunk_indices) + 1


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[str, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    return {path.stem: load_raw(path) for path in example_path.iterdir() if path.name.endswith('.png')}


def clip_T_world(xyzs_world, K, E, H, W):
    xyzs = torch.cat([xyzs_world, torch.ones_like(xyzs_world[..., 0:1, :])], dim=-2)
    K_expand = torch.zeros_like(E)
    znear, zfar = 1e-3, 1e3
    fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
    K_expand[:, 0, 0] = 2.0 * fx / W
    K_expand[:, 1, 1] = 2.0 * fy / H
    K_expand[:, 0, 2] = 2.0 * cx / W - 1.0
    K_expand[:, 1, 2] = 2.0 * cy / H - 1.0
    K_expand[:, 2, 2] = (zfar + znear) / (zfar - znear)
    K_expand[:, 3, 2] = 1.
    K_expand[:, 2, 3] = -2.0 * zfar * znear / (zfar - znear)
    return (K_expand @ E @ xyzs).permute(0, 2, 1)


def rasterize_lbs_weights(rasterize_context, xyz, lbs_weights, K, E, faces, resolution):
    xyz = torch.tensor(xyz).cuda()[None]
    lbs_weights = torch.tensor(lbs_weights).cuda()[None]
    K = torch.tensor(K).cuda()[None]
    E = torch.tensor(np.concatenate([E.reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)).cuda()[None]
    faces = torch.tensor(faces.astype(int)).cuda().type(torch.int32)
    NP = xyz.shape[1]

    resolution_new_0 = (resolution[0] // 8 + ((resolution[0] % 8) > 0)) * 8
    resolution_new_1 = (resolution[1] // 8 + ((resolution[1] % 8) > 0)) * 8

    xyzs_clip = clip_T_world(xyz.permute(0, 2, 1).float(), K.float(), E.float(), resolution_new_0,
                             resolution_new_1).contiguous()

    outputs, _ = nvdiffrast.torch.rasterize(rasterize_context, xyzs_clip, faces,
                                            [resolution_new_0, resolution_new_1])

    lbs_weights, _ = nvdiffrast.torch.interpolate(lbs_weights, outputs, faces, [resolution_new_0, resolution_new_1])
    return lbs_weights[0].contiguous().detach().cpu()


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, "camera"]
    cameras: Float[Tensor, "camera entry"]
    poses: Float[Tensor, "pose entry"]
    poses_tpose: Float[Tensor, "pose entry"]
    supervisions: Optional[Float[Tensor, "pose h w entry"]]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]
    # Optional: per-view face bbox [x1,y1,x2,y2] and confidence (from thuman_face_bboxes.json)
    face_bbox: Optional[Float[Tensor, "view 4"]]
    face_confidence: Optional[Float[Tensor, "view"]]


def load_metadata(camera_path: Path, canonical_path: Path, pose_path: Path, scene_name, split) -> Metadata:
    url = ""
    w = 1024
    h = 1024

    with open(camera_path, 'rb') as f:
        camera_infos = pickle.load(f)
    with open(canonical_path, 'rb') as f:
        canonical_infos = pickle.load(f)
    with open(pose_path, 'rb') as f:
        pose_infos = pickle.load(f)

    vertex = canonical_infos['vertex']
    lbs_weights = canonical_infos['weights']
    faces = canonical_infos['faces']

    timestamps = []
    cameras = []
    poses = []
    poses_tpose = []
    tposes_joints = []
    poses_angles_all = []
    lbs_weights_imgs = []

    if split == 'train':
        frame_orders = TRAIN_FRAME_ORDERS
    else:
        frame_orders = TEST_FRAME_ORDERS

    for i, frame in enumerate(frame_orders):
        key = f'frame_{frame:06d}'
        intrinsic = camera_infos[key]['intrinsics'].astype(np.float32)
        intrinsic = [intrinsic[0, 0] / w, intrinsic[1, 1] / h, intrinsic[0, 2] / w, intrinsic[1, 2] / h, 0.0, 0.0]
        w2c = camera_infos[key]['extrinsics'].astype(np.float32)
        Rh, Th = pose_infos[key]['Rh'], pose_infos[key]['Th']
        w2c = apply_global_tfm_to_camera(E=w2c, Rh=Rh, Th=Th)
        w2c = w2c[:3, :]
        w2c = w2c.flatten()

        camera = np.concatenate([intrinsic, w2c])
        cameras.append(camera)
        timestamps.append(frame)

        cnl_gtfms = get_canonical_global_tfms(pose_infos[key]['tpose_joints'], use_smplx=True)
        tpose_joints = pose_infos[key]['tpose_joints']
        poses_angles = pose_infos[key]['poses']

        dst_Rs, dst_Ts = body_pose_to_body_RTs(poses_angles, tpose_joints, use_smplx=True)
        global_Rs, global_Ts = get_global_RTs(cnl_gtfms, dst_Rs, dst_Ts, use_smplx=True)
        pose = np.concatenate([global_Rs.reshape(-1), global_Ts.reshape(-1)])

        dst_Rs_Tpose, dst_Ts_Tpose = body_pose_to_body_RTs(np.zeros_like(poses_angles), tpose_joints, use_smplx=True)
        global_Rs_Tpose, global_Ts_Tpose = get_global_RTs(cnl_gtfms, dst_Rs_Tpose, dst_Ts_Tpose, use_smplx=True)
        pose_tpose = np.concatenate([global_Rs_Tpose.reshape(-1), global_Ts_Tpose.reshape(-1)])

        poses.append(pose)
        poses_tpose.append(pose_tpose)
        tposes_joints.append(tpose_joints)
        poses_angles_all.append(poses_angles)

        if RASTERIZE_LBS_WEIGHTS:
            resolution = 1024
            rasterize_context = nvdiffrast.torch.RasterizeCudaContext(device='cuda')
            vertex_obs = apply_lbs_to_means(torch.tensor(vertex)[None], torch.tensor(global_Rs)[None],
                                   torch.tensor(global_Ts)[None], torch.tensor(lbs_weights)[None])
            vertex_obs = vertex_obs.detach().numpy()[0]
            intrinsics = camera_infos[key]['intrinsics'].astype(np.float32)
            intrinsics[:2] *= resolution / 1024
            lbs_weights_img = rasterize_lbs_weights(rasterize_context, vertex_obs, lbs_weights,
                                                    intrinsics, w2c, faces, [resolution, resolution])
            os.makedirs(os.path.join(OUTPUT_DIR, f"lbs_weights_supervisions", scene_name), exist_ok=True)
            path = os.path.join(OUTPUT_DIR, f"lbs_weights_supervisions", scene_name, key)
            np.savez_compressed(path, lbs_weights=lbs_weights_img.numpy())
            lbs_weights_imgs.append(lbs_weights_img)

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)
    poses = torch.tensor(np.stack(poses), dtype=torch.float32)
    poses_tpose = torch.tensor(np.stack(poses_tpose), dtype=torch.float32)
    tposes_joints = torch.tensor(np.stack(tposes_joints), dtype=torch.float32)
    poses_angles_all = torch.tensor(np.stack(poses_angles_all), dtype=torch.float32)
    
    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
        "poses": poses,
        "poses_tpose": poses_tpose,
        "tposes_joints": tposes_joints,
        "poses_angles_all": poses_angles_all,
    }


def process_single_key(key: str, path: Path, stage: str, pack_inconsistent: bool, output_dir: Path, face_bboxes: Optional[dict]):
    """Process a single key and return the example."""
    try:
        image_dir = path / key / "images"
        mask_dir = path / key / "masks"
        canonical_metafile = path / key / "canonical_joints.pkl"
        camera_metafile = path / key / "cameras.pkl"
        pose_metafile = path / key / "mesh_infos.pkl"
        
        if pack_inconsistent:
            ic_dir = Path("/workspace/humanvol/thuman_iclight") / stage / key / "images"

        # Read images and metadata
        images = load_images(image_dir)
        if pack_inconsistent:
            ic_images = load_images(ic_dir)
        masks = load_images(mask_dir)
        example = load_metadata(camera_metafile, canonical_metafile, pose_metafile, key, stage)

        num_bytes = get_size(path / key)
        
        # Merge the images into the example
        image_names = [f"frame_{timestamp.item():0>6}" for timestamp in example["timestamps"]]
        
        example["images"] = [images[image_name] for image_name in image_names]
        example["masks"] = [masks[image_name] for image_name in image_names]
        if pack_inconsistent:
            example["ic_images"] = [ic_images[image_name] for image_name in image_names]

        if face_bboxes is not None:
            try:
                face_detect_info = face_bboxes[key] # bbox, confidence scores, H,W
                bboxes = []
                confs = []
                for timestamp in example["timestamps"]:
                    padded_ts = f"{timestamp:06d}"
                    bboxes.append(face_detect_info[padded_ts]["bbox"])
                    confs.append(face_detect_info[padded_ts]["confidence"])
            except KeyError: # if none for the subject/key at all, return None
                example["face_bbox"] = None
                example["face_conf"] = None
                return example, num_bytes, None

            bboxes = [b if b is not None else [0., 0., 0., 0.] for b in bboxes]
            example["face_bbox"] = torch.tensor(bboxes)
            example["face_conf"] = torch.tensor(confs)

        # Explicitly delete the large dictionaries to free memory immediately
        del images
        del masks
        if pack_inconsistent:
            del ic_images
            
        assert len(example["images"]) == len(example["timestamps"])
        
        # Add the key to the example
        example["key"] = key
        
        return example, num_bytes, None
        
    except Exception as e:
        return None, 0, str(e)


def worker_process(worker_id: int, keys_queue: Queue, chunk_counter: Value, counter_lock: Lock,
                   stage: str, path: Path, pack_inconsistent: bool, output_dir: Path, 
                   target_bytes: int, progress_queue: Queue, face_bboxes: Optional[dict] = None):
    """Worker process that processes keys and saves chunks."""
    
    chunk_size = 0
    chunk = []
    local_index = {}
    
    def save_chunk():
        nonlocal chunk_size, chunk
        
        if not chunk:
            return
            
        # Atomically get next chunk index
        with counter_lock:
            chunk_idx = chunk_counter.value
            chunk_counter.value += 1
        
        chunk_key = f"{chunk_idx:0>6}"
        dir = output_dir / stage
        dir.mkdir(exist_ok=True, parents=True)
        
        chunk_path = dir / f"{chunk_key}.torch"
        torch.save(chunk, chunk_path)
        
        # Update local index
        for example in chunk:
            if "key" in example:
                local_index[example["key"]] = f"{chunk_key}.torch"
        
        progress_queue.put(('chunk_saved', worker_id, chunk_key, len(chunk), chunk_size))
        
        # Explicitly clear memory
        del chunk
        chunk = []
        chunk_size = 0
        gc.collect()  # Force garbage collection to free memory immediately
    
    # Process keys from queue
    while True:
        try:
            key = keys_queue.get(timeout=1)
            if key is None:  # Poison pill
                break
                
            example, num_bytes, error = process_single_key(key, path, stage, pack_inconsistent, output_dir, face_bboxes)
            
            if error:
                progress_queue.put(('error', worker_id, key, error))
                continue
            
            if example is None:
                progress_queue.put(('skipped', worker_id, key, "Unknown error"))
                continue
            
            chunk.append(example)
            chunk_size += num_bytes
            progress_queue.put(('processed', worker_id, key, num_bytes))
            
            if chunk_size >= target_bytes:
                save_chunk()
                
        except Exception as e:
            continue
    
    # Save remaining chunk
    if chunk_size > 0:
        save_chunk()
    
    # Send local index back
    progress_queue.put(('index', worker_id, local_index))
    progress_queue.put(('done', worker_id))


def categorize_keys_for_dryrun(all_keys: list[str], keys_to_force_reprocess: set[str], 
                                already_processed: set[str], force: bool) -> dict[str, list[str]]:
    """
    Categorize keys for dryrun display.
    
    Returns:
        Dictionary with categories as keys and lists of keys as values
    """
    categories = {
        'will_reprocess': [],  # Keys that will be reprocessed (from --reprocess-torch-files)
        'will_process_new': [],  # New keys that will be processed
        'will_skip_already_processed': [],  # Keys already processed (will be skipped)
        'not_found_in_torch': [],  # Keys in reprocess list but not found in torch files
    }
    
    if force:
        # In force mode, all keys will be processed
        categories['will_process_new'] = all_keys
        return categories
    
    for key in all_keys:
        if key in keys_to_force_reprocess:
            categories['will_reprocess'].append(key)
        elif key in already_processed:
            categories['will_skip_already_processed'].append(key)
        else:
            categories['will_process_new'].append(key)
    
    # Check for keys in reprocess list that weren't found
    if keys_to_force_reprocess:
        for key in keys_to_force_reprocess:
            if key not in all_keys:
                categories['not_found_in_torch'].append(key)
    
    return categories


def print_dryrun_summary(categories: dict[str, list[str]], stage: str, output_dir: Path):
    """Print a formatted dryrun summary."""
    print(f"\n{'='*80}")
    print(f"DRYRUN SUMMARY for stage: {stage}")
    print(f"Output directory: {output_dir / stage}")
    print(f"{'='*80}\n")
    
    total_keys = sum(len(keys) for keys in categories.values())
    
    # Will be reprocessed
    if categories['will_reprocess']:
        print(f"🔄 WILL BE REPROCESSED ({len(categories['will_reprocess'])} keys):")
        print(f"   (These keys are in --reprocess-torch-files and will be reprocessed)")
        for key in sorted(categories['will_reprocess'])[:20]:  # Show first 20
            print(f"   - {key}")
        if len(categories['will_reprocess']) > 20:
            print(f"   ... and {len(categories['will_reprocess']) - 20} more")
        print()
    
    # Will be processed (new)
    if categories['will_process_new']:
        print(f"✅ WILL BE PROCESSED (NEW) ({len(categories['will_process_new'])} keys):")
        print(f"   (These keys are not yet processed)")
        for key in sorted(categories['will_process_new'])[:20]:  # Show first 20
            print(f"   - {key}")
        if len(categories['will_process_new']) > 20:
            print(f"   ... and {len(categories['will_process_new']) - 20} more")
        print()
    
    # Will be skipped
    if categories['will_skip_already_processed']:
        print(f"⏭️  WILL BE SKIPPED (ALREADY PROCESSED) ({len(categories['will_skip_already_processed'])} keys):")
        print(f"   (These keys are already in the output directory)")
        for key in sorted(categories['will_skip_already_processed'])[:20]:  # Show first 20
            print(f"   - {key}")
        if len(categories['will_skip_already_processed']) > 20:
            print(f"   ... and {len(categories['will_skip_already_processed']) - 20} more")
        print()
    
    # Not found in torch files
    if categories['not_found_in_torch']:
        print(f"⚠️  WARNING: Keys in reprocess list but not found in torch files ({len(categories['not_found_in_torch'])} keys):")
        for key in sorted(categories['not_found_in_torch'])[:20]:
            print(f"   - {key}")
        if len(categories['not_found_in_torch']) > 20:
            print(f"   ... and {len(categories['not_found_in_torch']) - 20} more")
        print()
    
    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total keys in dataset: {total_keys}")
    print(f"  Will be reprocessed: {len(categories['will_reprocess'])}")
    print(f"  Will be processed (new): {len(categories['will_process_new'])}")
    print(f"  Will be skipped: {len(categories['will_skip_already_processed'])}")
    print(f"  Total to process: {len(categories['will_reprocess']) + len(categories['will_process_new'])}")
    print(f"{'='*80}\n")


def progress_monitor(progress_queue: Queue, total_keys: int, num_workers: int, 
                    interrupted_flag: threading.Event, index_path: Optional[Path] = None):
    """Monitor and display progress from all workers."""
    processed_count = 0
    error_count = 0
    skipped_count = 0
    chunks_saved = 0
    workers_done = 0
    total_bytes = 0
    all_indices = {}
    index_update_counter = 0
    
    pbar = tqdm(total=total_keys, desc="Processing keys")
    
    try:
        while True:
            try:
                msg = progress_queue.get(timeout=0.1)
                msg_type = msg[0]
                
                if msg_type == 'processed':
                    _, worker_id, key, num_bytes = msg
                    processed_count += 1
                    total_bytes += num_bytes
                    pbar.update(1)
                    pbar.set_postfix({
                        'chunks': chunks_saved,
                        'errors': error_count,
                        'MB': f"{total_bytes/1e6:.1f}"
                    })
                    
                elif msg_type == 'error':
                    _, worker_id, key, error = msg
                    error_count += 1
                    tqdm.write(f"Worker {worker_id}: Error processing {key}: {error}")
                    pbar.update(1)
                    
                elif msg_type == 'skipped':
                    _, worker_id, key, reason = msg
                    skipped_count += 1
                    tqdm.write(f"Worker {worker_id}: Skipped {key}: {reason}")
                    pbar.update(1)
                    
                elif msg_type == 'chunk_saved':
                    _, worker_id, chunk_key, chunk_len, chunk_size_bytes = msg
                    chunks_saved += 1
                    tqdm.write(f"Worker {worker_id}: Saved chunk {chunk_key} ({chunk_len} examples, {chunk_size_bytes/1e6:.2f} MB)")
                    
                elif msg_type == 'index':
                    _, worker_id, local_index = msg
                    all_indices.update(local_index)
                    index_update_counter += 1
                    
                    # Periodically write index to disk to free memory (every 10 worker completions)
                    if index_path and index_update_counter % 10 == 0:
                        try:
                            # Load existing index if it exists
                            existing_indices = {}
                            if index_path.exists():
                                with open(index_path, 'r') as f:
                                    existing_indices = json.load(f)
                            
                            # Merge with new indices
                            existing_indices.update(all_indices)
                            
                            # Write back to disk
                            with open(index_path, 'w') as f:
                                json.dump(existing_indices, f, indent=2)
                            
                            # Keep only a small working set in memory (last 100 entries)
                            # This prevents unbounded growth while still allowing updates
                            if len(all_indices) > 100:
                                # Keep only the most recent entries
                                recent_keys = list(all_indices.keys())[-100:]
                                all_indices = {k: all_indices[k] for k in recent_keys}
                                
                        except Exception as e:
                            tqdm.write(f"Warning: Could not update index periodically: {e}")
                    
                elif msg_type == 'done':
                    _, worker_id = msg
                    workers_done += 1
                    tqdm.write(f"Worker {worker_id} finished")
                    
            except KeyboardInterrupt:
                raise  # Re-raise KeyboardInterrupt
            except:
                # Check if interrupted flag is set
                if interrupted_flag.is_set():
                    break
                continue
                
            # Check if we're done (all workers finished)
            if workers_done >= num_workers:
                break
                
            # Check if interrupted
            if interrupted_flag.is_set():
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted in progress monitor!")
        raise
    finally:
        pbar.close()
    
    # If we've been doing periodic updates, reload the full index from disk
    if index_path and index_path.exists():
        try:
            with open(index_path, 'r') as f:
                disk_indices = json.load(f)
            # Merge any remaining in-memory indices
            disk_indices.update(all_indices)
            all_indices = disk_indices
        except Exception as e:
            print(f"Warning: Could not reload index from disk: {e}")
    
    return {
        'processed': processed_count,
        'errors': error_count,
        'skipped': skipped_count,
        'chunks': chunks_saved,
        'total_bytes': total_bytes,
        'index': all_indices
    }

if __name__ == "__main__":
        # Global variables for interrupt handling
    workers = []
    interrupted_flag = threading.Event()
    
    def signal_handler(sig, frame):
        print("\n\nInterrupt received! Terminating workers and exiting...")
        interrupted_flag.set()
        # Terminate all workers
        for p in workers:
            if p.is_alive():
                p.terminate()
        # Allow one more interrupt to force exit
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic", action="store_true")
    parser.add_argument("--face-bboxes", type=str, default="/workspace/humanvol/thuman/thuman_face_bboxes.json",
                        help="Path to thuman_face_bboxes.json (scene_id -> frame_000000 -> bbox, confidence). Adds face_bbox and face_confidence to each example.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--custom-json", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rasterize-lbs", action='store_true')
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--reprocess-torch-files", type=str, default=None,
                        help="Path to text file containing .torch filenames to reprocess (one per line)")
    parser.add_argument("--dryrun", action="store_true",
                        help="Dry run mode: show what would be processed without actually processing")
    args = parser.parse_args()
    
    RASTERIZE_LBS_WEIGHTS = args.rasterize_lbs
    pack_inconsistent = args.ic
    num_workers = args.num_workers
    
    if args.output_dir:
        print(f"Using output directory: {args.output_dir}")
        OUTPUT_DIR = Path(args.output_dir)
    else:
        print(f"Using default output directory: {OUTPUT_DIR}")

    print(f"pack_inconsistent: {pack_inconsistent}")
    print(f"num_workers: {num_workers}")
    if args.custom_json:
        print(f"Using custom JSON file: {args.custom_json}")
    if args.limit:
        print(f"Limiting to first {args.limit} entries")
    if args.force:
        print("Force mode: Will re-process all entries")
    if args.reprocess_torch_files:
        print(f"Reprocessing torch files listed in: {args.reprocess_torch_files}")
    if args.dryrun:
        print("DRYRUN MODE: No files will be processed or modified")

    try:
        for stage in ("train", "val"): # temp deleted train
            if interrupted_flag.is_set():
                print(f"\nSkipping remaining stages due to interrupt.")
                break
                
            print(f"\n{'='*60}")
            print(f"Processing stage: {stage}")
            print(f"{'='*60}\n")
            
            if stage == "train":
                path = INPUT_DIR / "train"
            elif stage == "val":
                path = INPUT_DIR / "val"
            elif stage == "test":
                path = INPUT_DIR / "val"
                
            keys = get_example_keys("train" if stage == "train" else "val",
                                    custom_json_path=args.custom_json,
                                    limit=args.limit)

            # Handle reprocessing of specific torch files
            torch_files_to_reprocess = []
            keys_to_force_reprocess = set()
            
            if args.reprocess_torch_files and not args.force:
                # Load the list of torch files to reprocess
                reprocess_file_path = Path(args.reprocess_torch_files)
                if reprocess_file_path.exists():
                    with open(reprocess_file_path, 'r') as f:
                        torch_files_to_reprocess = [line.strip() for line in f if line.strip()]
                    
                    print(f"\n{'='*60}")
                    print(f"Reprocessing mode: {len(torch_files_to_reprocess)} torch files")
                    print(f"{'='*60}")
                    
                    # Extract keys from these torch files
                    stage_path = OUTPUT_DIR / stage
                    if stage_path.exists():
                        keys_to_force_reprocess = get_keys_from_torch_files(torch_files_to_reprocess, stage_path)
                        
                        # Backup and delete the old torch files (skip in dryrun)
                        if not args.dryrun:
                            delete_torch_files(torch_files_to_reprocess, stage_path, backup=True)
                            print(f"Backed up and removed {len(torch_files_to_reprocess)} torch files")
                        else:
                            print(f"[DRYRUN] Would backup and remove {len(torch_files_to_reprocess)} torch files")
                    else:
                        print(f"Warning: Stage path {stage_path} does not exist. No files to reprocess.")
                else:
                    print(f"Warning: Reprocess file {args.reprocess_torch_files} not found. Skipping reprocessing.")

            # Resume functionality
            if args.force:
                already_processed = set()
                starting_chunk_index = 0
                print(f"Force mode: Processing all {len(keys)} entries from scratch.")
            elif keys_to_force_reprocess:
                # Exclude the torch files we're reprocessing from "already processed"
                already_processed = get_already_processed_keys(stage, OUTPUT_DIR, 
                                                              exclude_torch_files=torch_files_to_reprocess)
                
                # Filter keys: process only those that need reprocessing OR haven't been processed
                keys_needing_processing = []
                for key in keys:
                    if key in keys_to_force_reprocess or key not in already_processed:
                        keys_needing_processing.append(key)
                
                original_count = len(keys)
                keys = keys_needing_processing
                
                print(f"\nReprocessing {len(keys_to_force_reprocess)} keys from {len(torch_files_to_reprocess)} torch files")
                print(f"Plus {len(keys) - len(keys_to_force_reprocess)} new/unprocessed keys")
                print(f"Skipping {original_count - len(keys)} already-processed keys")
                
                starting_chunk_index = get_next_chunk_index(stage, OUTPUT_DIR)
                if starting_chunk_index > 0:
                    print(f"Continuing chunk numbering from index {starting_chunk_index}")
            else:
                already_processed = get_already_processed_keys(stage, OUTPUT_DIR)
                if already_processed:
                    original_count = len(keys)
                    keys = [k for k in keys if k not in already_processed]
                    skipped_count = original_count - len(keys)
                    print(f"Resuming: Skipping {skipped_count} already-processed entries. {len(keys)} remaining.")
                else:
                    print(f"Starting fresh: Processing {len(keys)} entries.")

                starting_chunk_index = get_next_chunk_index(stage, OUTPUT_DIR)
                if starting_chunk_index > 0:
                    print(f"Resuming: Starting from chunk index {starting_chunk_index}")

            if not keys:
                print(f"No keys to process for stage {stage}")
                continue

            # Dryrun mode: show what would be processed
            if args.dryrun:
                # Get all original keys for categorization
                all_original_keys = get_example_keys("train" if stage == "train" else "val",
                                                    custom_json_path=args.custom_json,
                                                    limit=args.limit)
                
                # Determine already_processed for categorization
                if args.force:
                    already_processed_for_dryrun = set()
                elif keys_to_force_reprocess:
                    # In dryrun, we still want to exclude the torch files from "already processed" 
                    # to show accurate categorization
                    already_processed_for_dryrun = get_already_processed_keys(stage, OUTPUT_DIR, 
                                                                            exclude_torch_files=torch_files_to_reprocess)
                else:
                    already_processed_for_dryrun = get_already_processed_keys(stage, OUTPUT_DIR)
                
                categories = categorize_keys_for_dryrun(all_original_keys, keys_to_force_reprocess, 
                                                        already_processed_for_dryrun, args.force)
                print_dryrun_summary(categories, stage, OUTPUT_DIR)
                continue  # Skip to next stage

            # Load face bboxes JSON if requested (for train split; keys must match scene ids)
            face_bboxes = None
            if args.face_bboxes and stage == "train":
                face_bbox_path = Path(args.face_bboxes)
                if face_bbox_path.exists():
                    with open(face_bbox_path) as f:
                        face_bboxes = json.load(f)
                    print(f"Loaded face bboxes for {len(face_bboxes)} scenes from {face_bbox_path}")
                else:
                    print(f"Warning: --face-bboxes file not found: {face_bbox_path}, skipping face bbox enrichment")

            # Set up multiprocessing
            mp.set_start_method('spawn', force=True)
            
            keys_queue = Queue()
            progress_queue = Queue()
            chunk_counter = Value('i', starting_chunk_index)
            counter_lock = Lock()
            
            # Fill the queue with keys
            for key in keys:
                if interrupted_flag.is_set():
                    break
                keys_queue.put(key)
            
            # Add poison pills for workers
            for _ in range(num_workers):
                keys_queue.put(None)
            
            # Start workers
            stage_workers = []
            try:
                for worker_id in range(num_workers):
                    if interrupted_flag.is_set():
                        break
                    p = mp.Process(
                        target=worker_process,
                        args=(worker_id, keys_queue, chunk_counter, counter_lock,
                              stage, path, pack_inconsistent, OUTPUT_DIR,
                              TARGET_BYTES_PER_CHUNK, progress_queue, face_bboxes)
                    )
                    p.start()
                    stage_workers.append(p)
                    workers.append(p)  # Also add to global list for signal handler
                
                # Monitor progress (pass index_path for periodic updates)
                stage_path = OUTPUT_DIR / stage
                index_path = stage_path / "index.json"
                stats = progress_monitor(progress_queue, len(keys), num_workers, interrupted_flag, index_path)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted! Terminating workers...")
                interrupted_flag.set()
                for p in stage_workers:
                    if p.is_alive():
                        p.terminate()
                raise
            
            # Wait for all workers with timeout
            for p in stage_workers:
                p.join(timeout=2.0)
                if p.is_alive():
                    print(f"Warning: Worker {p.pid} did not terminate cleanly, forcing termination...")
                    p.terminate()
                    p.join(timeout=1.0)
            
            if interrupted_flag.is_set():
                print("\n\nProcessing interrupted. Partial results may have been saved.")
                break
            
            print(f"\n{'='*60}")
            print(f"Stage {stage} complete:")
            print(f"  Processed: {stats['processed']}")
            print(f"  Errors: {stats['errors']}")
            print(f"  Skipped: {stats['skipped']}")
            print(f"  Chunks saved: {stats['chunks']}")
            print(f"  Total size: {stats['total_bytes']/1e6:.2f} MB")
            print(f"{'='*60}\n")
            
            # Save/update index
            print("Updating index...")
            stage_path = OUTPUT_DIR / stage
            index_path = stage_path / "index.json"
            
            # Load existing index if not force mode
            final_index = {}
            if index_path.exists() and not args.force:
                try:
                    with open(index_path, 'r') as f:
                        final_index = json.load(f)
                    print(f"Loaded {len(final_index)} existing entries from index.json")
                    
                    # If reprocessing, remove entries for the keys we just reprocessed
                    if keys_to_force_reprocess:
                        keys_removed = 0
                        for key in keys_to_force_reprocess:
                            if key in final_index:
                                del final_index[key]
                                keys_removed += 1
                        print(f"Removed {keys_removed} reprocessed keys from index")
                except Exception as e:
                    print(f"Warning: Could not load existing index: {e}")
            
            # Merge with new index
            final_index.update(stats['index'])
            
            # Save updated index
            if not interrupted_flag.is_set():
                # create index_path file if doesn't exist already
                if not index_path.exists():
                    index_path.touch()
                # then write to it
                with index_path.open("w") as f:
                    json.dump(final_index, f, indent=2)
                print(f"Index updated: {len(final_index)} total entries\n")
            
    except KeyboardInterrupt:
        print("\n\nForce interrupt received. Terminating all workers...")
        for p in workers:
            if p.is_alive():
                p.terminate()
        for p in workers:
            p.join(timeout=1.0)
        sys.exit(1)
    
    if interrupted_flag.is_set():
        print("\n\nProcessing interrupted by user. Partial results may have been saved.")
        sys.exit(130)