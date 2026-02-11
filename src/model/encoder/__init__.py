from typing import Optional

from .encoder import Encoder
from .encoder_noposplat import EncoderNoPoSplatCfg, EncoderNoPoSplat
from .encoder_noposplat_multi import EncoderNoPoSplatMulti
from .encoder_template_uv_concat_bone import EncoderTemplateUVConcatBone, EncoderLBSNoPoSplatCfg
from .encoder_template_uv_face import EncoderTemplateUVFace, EncoderLBSNoPoSplatFaceCfg
from .visualization.encoder_visualizer import EncoderVisualizer

ENCODERS = {
    "noposplat": (EncoderNoPoSplat, None),
    "noposplat_multi": (EncoderNoPoSplatMulti, None),
    "template_uv_concat_bone": (EncoderTemplateUVConcatBone, None),
    "template_uv_face": (EncoderTemplateUVFace, None),
}

EncoderCfg = EncoderNoPoSplatCfg | EncoderLBSNoPoSplatCfg | EncoderLBSNoPoSplatFaceCfg

def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
