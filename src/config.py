from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
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
}

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


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def separate_dataset_cfg_wrappers(joined: dict) -> list[DatasetCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: DatasetCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


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
