import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar, get_args

from dacite import Config, from_dict
from dacite.exceptions import UnionMatchError
from omegaconf import DictConfig, OmegaConf

from .dataset import DatasetCfgWrapper
from .dataset.data_module import DataLoaderCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.encoder.encoder_noposplat import EncoderNoPoSplatCfg
from .model.encoder.encoder_template_uv_concat_bone import EncoderLBSNoPoSplatCfg
from .model.encoder.encoder_template_uv_face import EncoderLBSNoPoSplatFaceCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    save_weights_only: bool


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_nodes: int = 1
    accumulate_grad_batches: int = 1


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: list[DatasetCfgWrapper]
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
    # YAML/CLI often give int (e.g. weight=0); dataclass expects float
    float: lambda x: float(x) if x is not None else None,
}

# Map loss config key (e.g. "faceloss") -> wrapper class for clearer UnionMatchError diagnostics
LOSS_KEY_TO_WRAPPER: dict[str, Type[LossCfgWrapper]] = {}
for _cls in get_args(LossCfgWrapper):
    _fields = dataclasses.fields(_cls)
    if len(_fields) == 1:
        LOSS_KEY_TO_WRAPPER[_fields[0].name] = _cls

# Same for dataset (e.g. "thuman") so we see the real nested error (e.g. view_sampler)
DATASET_KEY_TO_WRAPPER: dict[str, Type[DatasetCfgWrapper]] = {}
for _cls in get_args(DatasetCfgWrapper):
    _fields = dataclasses.fields(_cls)
    if len(_fields) == 1:
        DATASET_KEY_TO_WRAPPER[_fields[0].name] = _cls

# Map encoder name -> concrete config class so dacite can build encoder union
ENCODER_CFG_TYPES = {
    "noposplat": EncoderNoPoSplatCfg,
    "noposplat_multi": EncoderNoPoSplatCfg,
    "template_uv_concat_bone": EncoderLBSNoPoSplatCfg,
    "template_uv_face": EncoderLBSNoPoSplatFaceCfg,
}


def _encoder_cfg_from_dict(data: dict) -> EncoderCfg:
    """Build the correct encoder config by name so the union matches. Dacite calls type hooks with (data) only."""
    name = data.get("name")
    if name not in ENCODER_CFG_TYPES:
        raise ValueError(f"Unknown model.encoder name: {name!r}. Expected one of {list(ENCODER_CFG_TYPES)}")
    return from_dict(
        ENCODER_CFG_TYPES[name],
        data,
        config=Config(type_hooks=TYPE_HOOKS),
    )


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return  from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def separate_loss_cfg_wrappers(joined: dict | list) -> list[LossCfgWrapper]:
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    if isinstance(joined, list):
        items = []
        for x in joined:
            if isinstance(x, dict) and len(x) == 1:
                items.append(next(iter(x.items())))
            elif isinstance(x, dict) and x:
                items.append((next(iter(x.keys())), next(iter(x.values()))))
            elif isinstance(x, str):
                raise ValueError(
                    "loss is a list of names (e.g. CLI loss=[...]); need full loss dict. "
                    "Change losses in the experiment YAML, not loss= on the command line."
                ) from None
    else:
        items = list(joined.items()) if isinstance(joined, dict) else []

    out = []
    for k, v in items:
        if not isinstance(v, dict):
            if OmegaConf.is_config(v):
                v = OmegaConf.to_container(v, resolve=True)
            if not isinstance(v, dict):
                continue
        try:
            out.append(load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy)
        except UnionMatchError as e:
            # Parse with the specific wrapper for this key so the real error (e.g. wrong nested type) is raised
            if k in LOSS_KEY_TO_WRAPPER:
                try:
                    from_dict(
                        LOSS_KEY_TO_WRAPPER[k],
                        {k: v},
                        config=Config(type_hooks={**TYPE_HOOKS}),
                    )
                except Exception as inner:
                    raise RuntimeError(f"loss.{k}: {inner}") from inner
            msg = f"loss.{k}: {e}"
            if isinstance(v, dict):
                msg += f" (keys: {list(v.keys())})"
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"loss.{k}: {e}"
            if isinstance(v, dict):
                msg += f" (keys: {list(v.keys())})"
            raise RuntimeError(msg) from e
    return out


def separate_dataset_cfg_wrappers(joined: dict) -> list[DatasetCfgWrapper]:
    @dataclass
    class Dummy:
        dummy: DatasetCfgWrapper

    out = []
    for k, v in joined.items():
        if not isinstance(v, dict):
            if OmegaConf.is_config(v):
                v = OmegaConf.to_container(v, resolve=True)
            if not isinstance(v, dict):
                continue
        try:
            out.append(load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy)
        except UnionMatchError as e:
            if k in DATASET_KEY_TO_WRAPPER:
                try:
                    from_dict(
                        DATASET_KEY_TO_WRAPPER[k],
                        {k: v},
                        config=Config(type_hooks={**TYPE_HOOKS}),
                    )
                except Exception as inner:
                    raise RuntimeError(f"dataset.{k}: {inner}") from inner
            raise RuntimeError(f"dataset.{k}: {e} (keys: {list(v.keys()) if isinstance(v, dict) else 'n/a'})") from e
        except Exception as e:
            raise RuntimeError(f"dataset.{k}: {e}") from e
    return out


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {
            list[LossCfgWrapper]: separate_loss_cfg_wrappers,
            list[DatasetCfgWrapper]: separate_dataset_cfg_wrappers,
            EncoderCfg: _encoder_cfg_from_dict,
        },
    )
