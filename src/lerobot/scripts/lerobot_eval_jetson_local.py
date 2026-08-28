#!/usr/bin/env python
"""Evaluate LIBERO with actions served by a persistent Jetson-PI server.

This is intentionally independent from ``lerobot_eval.py``: it reuses the
original LIBERO environment and ``LiberoProcessorStep`` but does not load or
register a local LeRobot policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import numpy as np
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.io_utils import write_video


DEFAULT_CLIENT_PATH = Path(
    "/home/lhs/Works/Jetson-PI-Edge/bridge_local.py"
)
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"
TASK_IDS = tuple(range(10))


def load_client_module(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Jetson-PI client script not found: {path}")
    spec = importlib.util.spec_from_file_location("jetson_pi_libero_client", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Jetson-PI client script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    total_vit_ms = 0.0
    total_encode_ms = 0.0
    total_decode_ms = 0.0

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
        total_vit_ms += float(metadata.get("vit_ms") or 0.0)
        total_encode_ms += float(metadata.get("encode_ms") or 0.0)
        total_decode_ms += float(metadata.get("decode_ms") or 0.0)

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

    final_video_path = None
    if save_video_to is not None:
        outcome = str(success).lower()
        final_video_path = save_video_to.with_name(
            f"{save_video_to.stem}_{outcome}{save_video_to.suffix}"
        )
        final_video_path.parent.mkdir(parents=True, exist_ok=True)
        write_video(str(final_video_path), np.stack(frames), fps=int(env.unwrapped.metadata["render_fps"]))

    full_model_inference_ms = total_vit_ms + total_encode_ms + total_decode_ms
    average_model_inference_ms = full_model_inference_ms / inference_calls if inference_calls else 0.0

    return {
        "success": success,
        "steps": step,
        "inference_calls": inference_calls,
        "total_inference_ms": total_inference_ms,
        "total_vit_ms": total_vit_ms,
        "total_encode_ms": total_encode_ms,
        "total_decode_ms": total_decode_ms,
        "full_model_inference_ms": full_model_inference_ms,
        "average_model_inference_ms": average_model_inference_ms,
        "task": prompt,
        "seed": seed,
        "video": str(final_video_path) if final_video_path is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--client-path", type=Path, default=DEFAULT_CLIENT_PATH)
    parser.add_argument("--suite", default="libero_object")#libero_10,libero_spatial
    parser.add_argument("--episodes", type=int, default=1, help="episodes to run for each of the 10 tasks")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/jetson_pi_libero"))
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    client_module = load_client_module(args.client_path)
    client = client_module.JetsonPiLiberoClient(base_url=args.server_url)
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    results: list[dict[str, Any]] = []
    per_task_success_rate: dict[str, float] = {}

    for task_id in TASK_IDS:
        env_config = LiberoEnvConfig(
            task=args.suite,
            task_ids=[task_id],
            episode_length=args.episode_length,
            obs_type="pixels_agent_pos",
            control_mode="relative",
            init_states=True,
        )
        envs = make_env(env_config, n_envs=1, use_async_envs=False)
        env = envs[args.suite][task_id]
        task_results: list[dict[str, Any]] = []

        try:
            for episode in range(args.episodes):
                video_path = None
                if not args.no_video:
                    video_path = args.output_dir / "videos" / args.suite / f"task_{task_id}_episode_{episode}.mp4"
                result = run_episode(
                    env,
                    env_preprocessor,
                    client,
                    seed=args.seed + episode,
                    save_video_to=video_path,
                )
                result["task_id"] = task_id
                result["episode"] = episode
                task_results.append(result)
                results.append(result)
                print(json.dumps(result, ensure_ascii=False))
        finally:
            env.close()

        task_success_rate = float(np.mean([item["success"] for item in task_results]))
        per_task_success_rate[str(task_id)] = task_success_rate
        print(json.dumps({
            "suite": args.suite,
            "task_id": task_id,
            "episodes": args.episodes,
            "success_rate": task_success_rate,
        }, ensure_ascii=False))

    report = {
        "suite": args.suite,
        "task_ids": list(TASK_IDS),
        "episodes_per_task": args.episodes,
        "total_episodes": len(results),
        "success_rate": float(np.mean([item["success"] for item in results])),
        "per_task_success_rate": per_task_success_rate,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "eval_info.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "success_rate": report["success_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
