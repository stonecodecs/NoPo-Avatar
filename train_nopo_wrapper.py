#!/usr/bin/env python3
"""train_nopo_wrapper.py — Abstracted training launcher for NoPo-Avatar.

Composes the right base experiment YAML and dataset-level overrides from
high-level options, then invokes ``python -m src.main`` directly (no temp
YAML files).

Usage
-----
    python train_nopo_wrapper.py \
        --dataset THuman \
        --variations lighting crop \
        --ablations uvpe \
        -o batch_size=1 resolution=256 num_workers=2 \
        [--pretrained checkpoint/my.ckpt] \
        [--tag my_note] \
        [--dry-run]

Variations  (can be combined; 'all' expands to lighting+crop+varypose)
    none           No dataset augmentation — vanilla training
    lighting       Use inconsistent (relit/LBM) images for context views
    crop           Apply pre-computed bbox crops to context views
    varypose       Sample varying SMPLX poses per scene
    all            Shorthand for lighting+crop+varypose

Datasets
    THuman         THuman 2.1  (plain: thuman2.1_id_vtx, lighting: thuman2.1_lbm)
    XHuman         XHuman      (same dataset code as THuman, different root)
    MVHumanNet     MVHumanNet  (uses mvhn dataset config)

Ablations  (can be combined)
    none           Original NoPo-Avatar  (croco_multi2 backbone, no UV-PE, no face loss)
    face_encoder   Face encoder (DINOv2 FPN on face crop; unused now, but maybe a comeback in the future.)
    uvpe           UV positional encoding (croco_uv backbone, face loss enabled)
    sla            Structured local attention  [not yet implemented — reserved]
    all            Shorthand for uvpe + face_encoder (UV-PE + DINO face branch; SLA excluded until implemented)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent


# ─── Dataset Registry ─────────────────────────────────────────────────────────


@dataclass
class DatasetSpec:
    """Everything fixed per dataset."""

    hydra_group: str
    """The Hydra dataset group key, e.g. 'thuman' or 'mvhn'."""

    dataset_cfg_name: str
    """The YAML name inside config/dataset/, e.g. 'thuman' or 'mvhn'."""

    roots_plain: List[str]
    """Dataset roots when *no* lighting variant is active."""

    roots_lighting: List[str]
    """Dataset roots that contain relit/LBM images (may differ from plain)."""

    roots_varypose: Optional[List[str]] = None
    """Dataset roots for vary-pose sharding (if different from plain)."""

    has_crop_annotations: bool = False
    """Whether a crop_annotations.json exists for this dataset."""

    crop_annotations_path: Optional[str] = None
    """Repo-relative or absolute path to crop_annotations.json."""

    train_warning: Optional[str] = None
    """Printed before running if set (e.g. 'XHuman is typically a test-set')."""


DATASETS: dict[str, DatasetSpec] = {
    "THuman": DatasetSpec(
        hydra_group="thuman",
        dataset_cfg_name="thuman",
        roots_plain=["datasets/thuman2.1_lbm"],  # plain and lighting
        roots_lighting=["datasets/thuman2.1_lbm"],
        roots_varypose=["datasets/thuman2.1_id_vtx"],
        has_crop_annotations=False,
    ),
    "XHuman": DatasetSpec(
        hydra_group="thuman",          # reuses THuman dataset code
        dataset_cfg_name="thuman",
        roots_plain=["datasets/xhuman"],  # lighting (but can be used for plain training AND cropping)
        roots_lighting=["datasets/xhuman"],
        roots_varypose=["datasets/xhuman"],
        has_crop_annotations=True,  # if there are crop annotations (and selected), then this is the json to use.
        crop_annotations_path="crop_annotations.json",
        train_warning=(
            "XHuman is typically reserved as a test set. "
            "Training on it may leak evaluation data."
        ),
    ),
    # SO FAR, we don't use MVHumanNet yet. Need to complete its implementation
    # and by this, the only thing needs to be done is aligning the smplx to the image
    # (need to change camera coord frame).
    "MVHumanNet": DatasetSpec(
        hydra_group="mvhn",
        dataset_cfg_name="mvhn",
        roots_plain=["datasets/mvhn"],
        roots_lighting=["datasets/mvhn"],
        roots_varypose=["datasets/mvhn"],
        has_crop_annotations=False,
    ),
}


# ─── Ablation Registry ────────────────────────────────────────────────────────


@dataclass
class AblationSpec:
    """Model-architecture overrides for a given ablation."""

    description: str
    encoder: str
    backbone: str
    loss_list: List[str]
    extra_backbone_overrides: dict[str, str] = field(default_factory=dict)
    extra_model_overrides: dict[str, str] = field(default_factory=dict)


ABLATIONS: dict[str, AblationSpec] = {
    "none": AblationSpec(
        description="Original NoPo-Avatar (no UV-PE, no face loss)",
        encoder="template_uv_concat_bone",
        backbone="croco_multi2",
        loss_list=["mse", "lpips", "chamfer", "projection", "lbs_weights"],
        extra_backbone_overrides={"use_ref_mask": "true"},
    ), # not 'vanilla' since it uses refmask, but this makes more sense for 'training'
    "face_encoder": AblationSpec(
        description="Face encoder (DINOv2 FPN on face crop)",
        encoder="template_uv_face",
        backbone="croco_multi2",
        loss_list=["mse", "lpips", "chamfer", "projection", "lbs_weights", "faceloss"],
        extra_backbone_overrides={"use_ref_mask": "true"},
    ),
    "uvpe": AblationSpec(
        description="UV positional encoding (croco_uv backbone + face loss)",
        encoder="template_uv_face",
        backbone="croco_uv",
        loss_list=["mse", "lpips", "chamfer", "projection", "lbs_weights", "faceloss"],
        extra_backbone_overrides={
            "use_ref_mask": "true",
            "use_uv_pe": "true",
        },
    ),
    # Reserved — not yet implemented:
    "sla": AblationSpec(
        description="[NOT IMPLEMENTED] Structured local attention",
        encoder="template_uv_face",
        backbone="croco_uv",
        loss_list=["mse", "lpips", "chamfer", "projection", "lbs_weights", "faceloss"],
    ),
}


# ─── Base experiment matrix ───────────────────────────────────────────────────
# Maps (dataset_name, primary_ablation) → experiment YAML name.
# "Primary ablation" is the first non-'none' ablation, or 'none' if all are none.
# The experiment YAML sets image resolution and optimizer; everything else is
# overridden per-run.

# we build upon these experiments for the
BASE_EXPERIMENTS: dict[tuple[str, str], str] = {
    ("THuman",     "none"): "train_thuman2.0_simvs_3views_res256_face",  # 'none' means consistent set, but we still used this config
    ("THuman",     "uvpe"): "train_thuman2.0_simvs_3views_res256_face",  # this will be the main experiment for all else
    # not fully implemented yet, so below are TODO
    ("MVHumanNet", "none"): "train_mvhn_simvs_3views_res512",
    ("MVHumanNet", "uvpe"): "train_mvhn_simvs_3views_res256_face",
    # XHuman we don't really train on.
    ("XHuman",     "none"): "train_thuman2.0_3views_res1024",
    ("XHuman",     "uvpe"): "train_thuman2.0_simvs_3views_res512_face",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _hydra_list(values: list) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def _hydra_loss_list(losses: List[str]) -> str:
    """Hydra list override for loss; avoid spaces after commas."""
    return "[" + ",".join(losses) + "]"


def _append_ablation_spec(overrides: list[str], spec: AblationSpec) -> None:
    """Apply encoder, backbone, loss, and backbone extras from an AblationSpec.

    Use Hydra *config-group* paths (slashes) for encoder and backbone so the
    nested YAMLs are composed. ``model.encoder=name`` would replace the node in
    a way that drops ``backbone`` and breaks ``model.encoder.backbone=...``.
    """
    overrides.append(f"model/encoder={spec.encoder}")
    overrides.append(f"model/encoder/backbone={spec.backbone}")
    overrides.append(f"loss={_hydra_loss_list(spec.loss_list)}")
    for k, v in spec.extra_backbone_overrides.items():
        overrides.append(f"model.encoder.backbone.{k}={v}")
    for k, v in spec.extra_model_overrides.items():
        overrides.append(f"model.encoder.{k}={v}")


def _deduplicate_run_name(name: str) -> str:
    """Return `name` or `name_v{N}` if outputs/exp_<name> already has prior runs."""
    base_dir = REPO_ROOT / "outputs" / f"exp_{name}"
    if not base_dir.exists() or not any(base_dir.iterdir()):
        return name
    # Find the next free vN suffix.
    n = 2
    while True:
        candidate = f"{name}_v{n}"
        cdir = REPO_ROOT / "outputs" / f"exp_{candidate}"
        if not cdir.exists() or not any(cdir.iterdir()):
            print(
                f"[train_nopo] outputs/exp_{name} already exists — using run name '{candidate}'."
            )
            return candidate
        n += 1


# ─── Config / override builder ────────────────────────────────────────────────


def build_overrides(
    dataset_name: str,
    variations: list[str],
    ablations: list[str],
    pretrained: Optional[str],
    run_name: str,
    extra_overrides: list[str],
) -> tuple[str, list[str]]:
    """
    Returns ``(base_experiment_yaml_name, list_of_hydra_overrides)``.

    The base experiment YAML is responsible for the Hydra defaults list
    (encoder, backbone, decoder, loss).  We inject *runtime* dataset/wandb
    overrides and apply ``ABLATIONS[primary_ablation]`` (encoder, backbone,
    loss, backbone flags) so the same base YAML can match ``none`` vs ``uvpe``.
    """
    dataset = DATASETS[dataset_name]
    dkey = dataset.hydra_group  # "thuman" or "mvhn"

    # Primary ablation selects which base experiment YAML to use.
    primary_ablation = next((a for a in ablations if a != "none"), "none")

    base_exp = BASE_EXPERIMENTS[(dataset_name, primary_ablation)]

    # ── Dataset root: use lighting roots if lighting variant is active ─────────
    use_lighting = "lighting" in variations
    use_varypose = "varypose" in variations
    if use_varypose:  # if varypose and lighting, it uses varypose as priority
        roots = dataset.roots_varypose or dataset.roots_plain
    elif use_lighting:
        roots = dataset.roots_lighting
    else:
        roots = dataset.roots_plain

    overrides: list[str] = [
        # ── mode ──────────────────────────────────────────────────────────────
        "mode=train",
        # ── dataset root & variation flags ────────────────────────────────────
        f"dataset.{dkey}.roots={_hydra_list(roots)}",
        f"dataset.{dkey}.load_inconsistent_images={str(use_lighting).lower()}",
        f"dataset.{dkey}.vary_poses={'true' if use_varypose else 'false'}",
        # ── wandb identity ────────────────────────────────────────────────────
        f"wandb.name=nopo_train_{run_name}",
    ]
    tags: list[str] = [dkey]

    # Architecture / loss from ABLATIONS (same base experiment YAML can differ by primary ablation).
    if primary_ablation in ABLATIONS:
        _append_ablation_spec(overrides, ABLATIONS[primary_ablation])

    if "face_encoder" not in ablations:
        # unless explicitly stated, we don't use face encoder
        # this is needed since template_uv_face defaults to using it.
        overrides.append("model.encoder.face_encoder_cfg=null")

    # ── Crop annotations ──────────────────────────────────────────────────────
    if "crop" in variations:
        if not dataset.has_crop_annotations or not dataset.crop_annotations_path:
            print(
                f"[train_nopo] WARNING: 'crop' variation requested but "
                f"no crop_annotations configured for {dataset_name}. Skipping crop."
            )
            overrides.append(f"dataset.{dkey}.crop_annotations_path=null")
        else:
            crop_path = Path(dataset.crop_annotations_path)
            if not crop_path.is_absolute():
                crop_path = REPO_ROOT / crop_path
            overrides.append(f"dataset.{dkey}.crop_annotations_path={crop_path}")
    else:
        overrides.append(f"dataset.{dkey}.crop_annotations_path=null")

    # ── Pretrained weights ────────────────────────────────────────────────────
    if pretrained:
        overrides.append(f"model.encoder.pretrained_weights={pretrained}")

    # ── User-supplied extra overrides ─────────────────────────────────────────
    for ovr in extra_overrides:
        # for some common ones, we can simplify the syntax
        key, val = ovr.split("=")
        if key == "batch_size":
            overrides.append(f"data_loader.train.{key}={val}")
        elif key == "resolution" or key == "res":
            # keep template resolution the same (1024x1024) and only change the input shape
            # these all need to match
            overrides.append(f"dataset.{dkey}.input_image_shape=[{val}, {val}]")
            overrides.append(f"dataset.{dkey}.template_image_shape=[{val}, {val}]")
            overrides.append(f"model.encoder.backbone.template_image_size=[{val}, {val}]")
            tags.append(f"{val}x{val}")
        elif key == "num_workers":
            overrides.append(f"data_loader.train.{key}={val}")
        elif key == "consistent_set_prob":
            overrides.append(f"dataset.{dkey}.consistent_set_prob={val}")
        elif key == "output_path":
            overrides.append(f"hydra.run.dir={val}")
        elif key == "checkpoint":
            overrides.append(f"checkpointing.load={val}")
        else:
            overrides.append(ovr)
    overrides.append(f"wandb.tags={tags}")
    return base_exp, overrides


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Abstracted training launcher for NoPo-Avatar experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        choices=list(DATASETS.keys()),
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--variations", "-v",
        nargs="+",
        default=["none"],
        help=(
            "Dataset variations to activate. Choose from: "
            "none, lighting, crop, varypose, all. "
            "Can be combined (e.g. --variations lighting crop)."
        ),
    )
    parser.add_argument(
        "--ablations", "-a",
        nargs="+",
        default=["none"],
        choices=["none", "uvpe", "sla", "face_encoder", "all"],
        help="Model ablations to enable (none = original NoPo-Avatar).",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained encoder weights (.ckpt) to warm-start from.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional suffix appended to the auto-generated run name.",
    )
    parser.add_argument(
        "--override", "-o",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Raw Hydra overrides forwarded verbatim, e.g. trainer.max_steps=100000.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be run without actually executing it.",
    )
    args = parser.parse_args()

    # ── Expand 'all' variation ────────────────────────────────────────────────
    variations: list[str] = []
    for v in args.variations:
        if v == "all":
            variations.extend(["lighting", "crop", "varypose"])
        elif v == "none":
            pass  # empty list → no variations
        else:
            variations.append(v)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    variations = [v for v in variations if not (v in seen or seen.add(v))]

    # decompose 'all' into implemented components (uvpe + face DINO; SLA reserved)
    if "all" in args.ablations:
        args.ablations = ["uvpe", "face_encoder"]

    # ── SLA guard ─────────────────────────────────────────────────────────────
    if "sla" in args.ablations:
        print("[train_nopo] ERROR: 'sla' ablation is not yet implemented. Omitting.", file=sys.stderr)
        args.ablations.remove("sla")

    # ── Dataset warning ───────────────────────────────────────────────────────
    dataset_spec = DATASETS[args.dataset]
    if dataset_spec.train_warning:
        print(f"[train_nopo] WARNING: {dataset_spec.train_warning}")
        response = input("Continue anyway? [y/N] ").strip().lower()
        if response != "y":
            print("[train_nopo] Aborted.")
            sys.exit(0)

    # ── Generate run name ─────────────────────────────────────────────────────
    parts = [args.dataset.lower()]
    if variations:
        parts.append("-".join(variations))   # use '-' not '+' (Hydra treats '+' specially)
    else:
        parts.append("vanilla")
    for ablation in args.ablations:
        if ablation != "none":
            parts.append(ablation)
    if args.tag:
        parts.append(args.tag)
    base_run_name = "_".join(parts)

    if not args.dry_run:
        run_name = _deduplicate_run_name(base_run_name)
    else:
        run_name = base_run_name

    # ── Build overrides ───────────────────────────────────────────────────────
    base_exp, overrides = build_overrides(
        dataset_name=args.dataset,
        variations=variations,
        ablations=args.ablations,
        pretrained=args.pretrained,
        run_name=run_name,
        extra_overrides=args.override,
    )

    # ── Assemble subprocess command ───────────────────────────────────────────
    # Use +experiment= to compose base architecture, then layer all overrides.
    cmd = [
        sys.executable, "-m", "src.main",
        f"+experiment={base_exp}",
        *overrides,
    ]

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    if "varypose" in variations:
        active_roots = dataset_spec.roots_varypose or dataset_spec.roots_plain
    elif "lighting" in variations:
        active_roots = dataset_spec.roots_lighting
    else:
        active_roots = dataset_spec.roots_plain
    print(f"  Run name   : {run_name}")
    print(f"  Dataset    : {args.dataset}  (roots: {active_roots})")
    print(f"  Variations : {variations if variations else ['none']}")
    print(f"  Ablations  : {args.ablations}")
    print(f"  Base YAML  : +experiment={base_exp}")
    print(f"  Pretrained : {args.pretrained or '(from base YAML)'}")
    print(f"{'─' * 72}")
    print(f"\n  CMD: {' '.join(cmd)}\n")

    if args.dry_run:
        print("[train_nopo] --dry-run: not executing.")
        return

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
