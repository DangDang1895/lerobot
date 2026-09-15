# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference engine configs and factory.

Selection is explicit via ``--inference.type=sync|rtc|jetson_pi``.  Adding a new
backend requires registering its config subclass and dispatching it in
:func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from threading import Event
from typing import Any

import draccus

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine
from .jetson_pi import JetsonPiSyncInferenceEngine
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for inference backend configuration.

    Use ``--inference.type=<name>`` on the CLI to select a backend.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30


@InferenceEngineConfig.register_subclass("jetson_pi")
@dataclass
class JetsonPiInferenceConfig(InferenceEngineConfig):
    """Inline inference through the Jetson-PI-Edge pybind11 module.

    ``--policy.path`` remains required for policy metadata, feature names,
    state normalization and action unnormalization; Torch weights are not loaded.
    Policy text queries are not supported, so ``/vqa`` and ``/autosteer`` are unavailable.
    """

    model_path: str
    mmproj_path: str
    module_path: str
    backend: str = "cuda"
    image_keys: tuple[str, ...] = ("global", "wrist")
    image_height: int = 224
    image_width: int = 224
    state_dim: int = 32
    action_steps: int = 50
    action_dim: int = 32
    chunk_size: int = 50
    n_threads: int = 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy | None,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    robot_wrapper: ThreadSafeRobot,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: Event | None = None,
    jetson_pi_model: Any | None = None,
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object."""
    logger.info("Creating inference engine: %s", config.type)
    if isinstance(config, JetsonPiInferenceConfig):
        if jetson_pi_model is None:
            raise ValueError("Jetson-PI model must be loaded before creating its inference engine")
        return JetsonPiSyncInferenceEngine(
            model=jetson_pi_model,
            backend=config.backend,
            image_keys=config.image_keys,
            image_height=config.image_height,
            image_width=config.image_width,
            state_dim=config.state_dim,
            action_steps=config.action_steps,
            action_dim=config.action_dim,
            chunk_size=config.chunk_size,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            robot_type=robot_wrapper.robot_type,
        )
    if policy is None:
        raise ValueError(f"Inference backend {config.type!r} requires a loaded Torch policy")
    if isinstance(config, SyncInferenceConfig):
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
        )
    if isinstance(config, RTCInferenceConfig):
        return RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
