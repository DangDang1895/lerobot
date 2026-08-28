#!/usr/bin/env python
"""Evaluate LIBERO with actions served by a persistent Jetson-PI server.

This is intentionally independent from ``lerobot_eval.py``: it reuses the
original LIBERO environment and ``LiberoProcessorStep`` but does not load or
register a local LeRobot policy.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gymnasium as gym
import numpy as np
from PIL import Image
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.io_utils import write_video


IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"


def _image_as_png_base64(image: Any) -> str:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"expected a 3-D image, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"expected an RGB image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("image contains non-finite values")
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
    array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array, mode="RGB")
    if pil_image.size != (224, 224):
        pil_image = pil_image.resize((224, 224), Image.Resampling.BILINEAR)
    output = io.BytesIO()
    pil_image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


class RemoteJetsonPiClient:
    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def predict(
        self,
        images: list[Any],
        state: Any,
        prompt: str,
        *,
        reset: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if len(images) != 2:
            raise ValueError(f"expected exactly two images, got {len(images)}")
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.shape != (8,) or not np.isfinite(state_array).all():
            raise ValueError(f"expected a finite 8-D state, got {state_array.shape}")
        payload = {
            "images": [_image_as_png_base64(image) for image in images],
            "state": state_array.tolist(),
            "prompt": prompt,
            "reset": bool(reset),
        }
        request = Request(
            self.base_url + "/infer",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"remote Jetson-PI request failed ({error.code}): {detail}") from error
        except URLError as error:
            raise RuntimeError(f"could not connect to Jetson-PI bridge at {self.base_url}: {error}") from error
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"bridge returned invalid actions: {actions.shape}")
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return np.ascontiguousarray(actions), metadata


def _one_batch_item(value: Any, name: str) -> Any:
    if not hasattr(value, "shape") or len(value.shape) == 0 or value.shape[0] != 1:
        raise ValueError(f"{name} must have batch size 1, got {getattr(value, 'shape', None)}")
    return value[0]


def extract_model_inputs(observation: dict[str, Any]) -> tuple[list[Any], Any]:
    missing = [key for key in (*IMAGE_KEYS, STATE_KEY) if key not in observation]
    if missing:
        raise KeyError(f"processed LIBERO observation is missing: {missing}")
    images = [_one_batch_item(observation[key], key) for key in IMAGE_KEYS]
    state = _one_batch_item(observation[STATE_KEY], STATE_KEY)
    return images, state


def _render_one(env: gym.vector.VectorEnv) -> np.ndarray:
    if isinstance(env, gym.vector.SyncVectorEnv):
        return np.asarray(env.envs[0].render())
    return np.asarray(env.call("render")[0])


def _success_from_info(info: dict[str, Any]) -> bool:
    if "is_success" in info:
        value = np.asarray(info["is_success"]).reshape(-1)
        return bool(value[0]) if value.size else False
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict) and "is_success" in final_info:
            value = np.asarray(final_info["is_success"]).reshape(-1)
            return bool(value[0]) if value.size else False
        if len(final_info) and isinstance(final_info[0], dict):
            return bool(final_info[0].get("is_success", False))
    return False


def run_episode(
    env: gym.vector.VectorEnv,
    env_preprocessor: Any,
    client: Any,
    *,
    seed: int,
    save_video_to: Path | None,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=[seed])
    prompt = str(env.call("task_description")[0])
    max_steps = int(env.call("_max_episode_steps")[0])
    frames = [_render_one(env)] if save_video_to is not None else []
    step = 0
    inference_calls = 0
    success = False
    total_inference_ms = 0.0

    while step < max_steps and not success:
        processed = env_preprocessor(preprocess_observation(observation))
        images, state = extract_model_inputs(processed)
        action_chunk, metadata = client.predict(
            images,
            state,
            prompt,
            reset=inference_calls == 0,
        )
        if action_chunk.shape != (10, 7):
            raise RuntimeError(f"client returned {action_chunk.shape}, expected (10, 7)")
        inference_calls += 1
        total_inference_ms += float(metadata.get("total_ms") or 0.0)

        for action in action_chunk:
            if step >= max_steps or success:
                break
            observation, _, terminated, truncated, info = env.step(action[None, :])
            step += 1
            if save_video_to is not None:
                frames.append(_render_one(env))
            success = _success_from_info(info)
            if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                break

    if save_video_to is not None:
        save_video_to.parent.mkdir(parents=True, exist_ok=True)
        write_video(str(save_video_to), np.stack(frames), fps=int(env.unwrapped.metadata["render_fps"]))

    return {
        "success": success,
        "steps": step,
        "inference_calls": inference_calls,
        "total_inference_ms": total_inference_ms,
        "task": prompt,
        "seed": seed,
        "video": str(save_video_to) if save_video_to is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://10.32.220.227:1896")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/jetson_pi_libero"))
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    client = RemoteJetsonPiClient(base_url=args.server_url, timeout=args.timeout)
    env_config = LiberoEnvConfig(
        task=args.suite,
        task_ids=[args.task_id],
        episode_length=args.episode_length,
        obs_type="pixels_agent_pos",
        control_mode="relative",
        init_states=True,
    )
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs[args.suite][args.task_id]
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    results: list[dict[str, Any]] = []

    try:
        for episode in range(args.episodes):
            video_path = None
            if not args.no_video:
                video_path = args.output_dir / "videos" / args.suite / f"task_{args.task_id}_episode_{episode}.mp4"
            result = run_episode(
                env,
                env_preprocessor,
                client,
                seed=args.seed + episode,
                save_video_to=video_path,
            )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    finally:
        env.close()

    report = {
        "suite": args.suite,
        "task_id": args.task_id,
        "episodes": args.episodes,
        "success_rate": float(np.mean([item["success"] for item in results])),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "eval_info.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "success_rate": report["success_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
