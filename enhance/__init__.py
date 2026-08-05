"""Composable enhancement stages for the Real-ESRGAN video runner."""

import sys

from . import basicvsrpp_autodl as _basicvsrpp_autodl

# The AutoDL branch targets a single high-end CUDA GPU. Keep the public import
# path stable while routing only this branch through the CUDA-resident reduction
# path. The Kaggle branch remains fixed at commit 3.6.
sys.modules[f"{__name__}.basicvsrpp"] = _basicvsrpp_autodl

from .basicvsrpp_autodl import (
    BASICVSRPP_TRACK_URLS,
    BasicVSRPPConfig,
    BasicVSRPPPreprocessor,
    BasicVSRPPStreamReader,
    BasicVSRPlusPlusNet,
)
from .pipeline import FrameEnhancementPipeline, PipelineConfig
from .srvgg import SRVGGNetCompact, assert_no_extra_state

__all__ = [
    "BASICVSRPP_TRACK_URLS",
    "BasicVSRPPConfig",
    "BasicVSRPPPreprocessor",
    "BasicVSRPPStreamReader",
    "BasicVSRPlusPlusNet",
    "FrameEnhancementPipeline",
    "PipelineConfig",
    "SRVGGNetCompact",
    "assert_no_extra_state",
]
