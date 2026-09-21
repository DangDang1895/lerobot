# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Synchronous in-process inference through the Jetson-PI-Edge pybind module."""

from __future__ import annotations

import importlib
import json
import logging
import sys
from collections import deque
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.policies.common.vla_utils import resize_with_pad_torch
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

from .base import InferenceEngine

logger = logging.getLogger(__name__)


def _load_processor_step(
    root: Path,
    config_filename: str,
    registry_name: str,
    processor_type: type,
    config_overrides: dict[str, Any] | None = None,
) -> Any:
    """Load one processor step and its state from a policy checkpoint."""
    config_path = root / config_filename
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entries = [entry for entry in config["steps"] if entry.get("registry_name") == registry_name]
    if len(entries) != 1:
        raise ValueError(f"Expected one {registry_name} step in {config_path}, found {len(entries)}")

    entry = entries[0]
    processor_config = {**entry.get("config", {}), **(config_overrides or {})}
    processor = processor_type(**processor_config)
    state_file = entry.get("state_file")
    if state_file:
        processor.load_state_dict(load_file(str(root / state_file)))
    return processor


def make_jetson_pi_processors(
    pretrained_path: str | Path,
    rename_map: dict[str, str] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Load only state normalization and action unnormalization.

    Jetson-PI-Edge already implements PI0.5 prompt formatting, tokenization and
    model execution. Running LeRobot's complete policy preprocessor here would
    format/tokenize the request twice.
    """
    root = Path(pretrained_path).expanduser().resolve()
    rename_processor = _load_processor_step(
        root,
        "policy_preprocessor.json",
        "rename_observations_processor",
        RenameObservationsProcessorStep,
        config_overrides={"rename_map": rename_map or {}},
    )
    normalizer = _load_processor_step(
        root,
        "policy_preprocessor.json",
        "normalizer_processor",
        NormalizerProcessorStep,
    )
    unnormalizer = _load_processor_step(
        root,
        "policy_postprocessor.json",
        "unnormalizer_processor",
        UnnormalizerProcessorStep,
    )
    preprocessor = PolicyProcessorPipeline(
        steps=[rename_processor, normalizer],
        name="jetson_pi_preprocessor",
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=[unnormalizer],
        name="jetson_pi_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def load_jetson_pi_model(
    *,
    model_path: str,
    mmproj_path: str,
    module_path: str,
    backend: str,
    n_views: int,
    image_height: int,
    image_width: int,
    n_threads: int,
) -> Any:
    """Import the external pybind module and load its model before hardware connects."""
    module_dir = Path(module_path).expanduser().resolve()
    if not module_dir.is_dir():
        raise FileNotFoundError(f"Jetson-PI Python module directory does not exist: {module_dir}")
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)
    try:
        jetson_pi = importlib.import_module("jetson_pi")
    except ImportError as exc:
        raise ImportError(
            f"Could not import the Jetson-PI pybind module from {module_dir}. "
            "Build it with the same Python version used by lerobot-rollout."
        ) from exc

    return jetson_pi.load_model(
        model_path=model_path,
        mmproj_path=mmproj_path,
        backend=backend,
        n_views=n_views,
        image_height=image_height,
        image_width=image_width,
        n_threads=n_threads,
    )


def _prepare_image(image: Any, image_height: int, image_width: int) -> np.ndarray:
    """Resize one RGB image with PI0.5 padding and return contiguous uint8 HWC."""
    image = torch.as_tensor(image).detach().cpu()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected image batch size 1, got shape {tuple(image.shape)}")
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D image tensor, got shape {tuple(image.shape)}")
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    elif image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {tuple(image.shape)}")

    image = image.to(torch.float32)
    if image.shape[:2] != (image_height, image_width):
        image = resize_with_pad_torch(image, image_height, image_width)
    if image.ndim == 4:
        image = image[0]
    if image.numel() and image.max() > 1:
        image = image / 255
    image = torch.round(image.clamp(0, 1) * 255).to(torch.uint8)
    return np.ascontiguousarray(image.numpy())


def _prepare_state(state: Any, state_dim: int) -> np.ndarray:
    """Flatten normalized robot state and zero-pad it to the model dimension."""
    state = torch.as_tensor(state, dtype=torch.float32).detach().cpu()
    if state.ndim == 2 and state.shape[0] == 1:
        state = state[0]
    state_array = np.asarray(state.reshape(-1).numpy(), dtype=np.float32)
    if state_array.size > state_dim:
        raise ValueError(f"Robot state dimension {state_array.size} exceeds model dimension {state_dim}")
    if not np.isfinite(state_array).all():
        raise ValueError("State contains non-finite values")
    return np.pad(state_array, (0, state_dim - state_array.size)).astype(np.float32, copy=False)


class JetsonPiSyncInferenceEngine(InferenceEngine):
    """Generate PI0.5 chunks in process and dispatch them synchronously."""

    def __init__(
        self,
        *,
        model: Any,
        backend: str,
        image_keys: tuple[str, ...],
        image_height: int,
        image_width: int,
        state_dim: int,
        action_steps: int,
        action_dim: int,
        chunk_size: int,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        robot_type: str,
    ) -> None:
        super().__init__(task=task)
        if not image_keys:
            raise ValueError("Jetson-PI requires at least one image key")
        if not 0 < chunk_size <= action_steps:
            raise ValueError(f"chunk_size must be in [1, {action_steps}], got {chunk_size}")
        if len(ordered_action_keys) > action_dim:
            raise ValueError(
                f"Robot action dimension {len(ordered_action_keys)} exceeds model dimension {action_dim}"
            )

        self._model = model
        self._image_keys = tuple(
            key if key.startswith("observation.images.") else f"observation.images.{key}"
            for key in image_keys
        )
        self._image_height = image_height
        self._image_width = image_width
        self._state_dim = state_dim
        self._action_steps = action_steps
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._robot_type = robot_type
        self._action_queue: deque[tuple[np.ndarray, str]] = deque()
        logger.info(
            "JetsonPiSyncInferenceEngine initialized (backend=%s, views=%d, chunk_size=%d)",
            backend,
            len(image_keys),
            chunk_size,
        )

    @property
    def control_thread_owns_policy(self) -> bool:
        return True

    def start(self) -> None:
        logger.info("JetsonPiSyncInferenceEngine started (inline pybind mode)")

    def stop(self) -> None:
        self._action_queue.clear()
        if self._model is not None:
            self._model.close()
            self._model = None
        logger.info("JetsonPiSyncInferenceEngine stopped")

    def reset(self) -> None:
        self._action_queue.clear()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._discard_task_change()
        logger.info("Reset Jetson-PI action queue and processors")

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if obs_frame is None:
            return None

        task, task_changed = self._take_task()
        if task_changed:
            logger.info("Task changed to '%s' - dropping precomputed Jetson-PI actions", task)
            self._action_queue.clear()
        if not self._action_queue:
            self._request_actions(obs_frame, task)

        normalized_action, action_task = self._action_queue.popleft()
        action = torch.from_numpy(normalized_action).to(torch.float32).unsqueeze(0)
        action = self._postprocessor(action).squeeze(0).cpu()
        action_dict = make_robot_action(action, self._dataset_features)
        self._set_dispatched_task(action_task)
        return torch.tensor([action_dict[key] for key in self._ordered_action_keys])

    def _request_actions(self, obs_frame: dict, task: str) -> None:
        observation = prepare_observation_for_inference(
            copy(obs_frame), torch.device("cpu"), task, self._robot_type
        )
        observation = self._preprocessor(observation)

        missing = [key for key in self._image_keys if key not in observation]
        if missing:
            raise KeyError(f"Missing Jetson-PI image observations: {missing}")
        images = np.ascontiguousarray(
            np.stack(
                [
                    _prepare_image(observation[key], self._image_height, self._image_width)
                    for key in self._image_keys
                ]
            ),
            dtype=np.uint8,
        )
        # PI0.5 encodes state in text: preserve the checkpoint's semantic width.
        state_width = torch.as_tensor(observation["observation.state"]).numel()
        if state_width > self._state_dim:
            raise ValueError(f"Robot state dimension {state_width} exceeds model dimension {self._state_dim}")
        state = _prepare_state(observation["observation.state"], state_width)
        prompt = task.strip().replace("_", " ").replace("\n", " ")
        if not prompt:
            raise ValueError("Task prompt must not be empty")

        actions = np.asarray(self._model.predict(images, prompt, state), dtype=np.float32)
        expected_shape = (self._action_steps, self._action_dim)
        if actions.shape != expected_shape:
            raise RuntimeError(f"Expected Jetson-PI action shape {expected_shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("Jetson-PI returned non-finite actions")

        robot_action_dim = len(self._ordered_action_keys)
        for action in actions[: self._chunk_size, :robot_action_dim]:
            self._action_queue.append((np.ascontiguousarray(action), task))
