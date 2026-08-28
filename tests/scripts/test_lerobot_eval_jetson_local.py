import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "src/lerobot/scripts/lerobot_eval_jetson_local.py"


def load_module():
    spec = importlib.util.spec_from_file_location("lerobot_eval_jetson_local", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_no_longer_exposes_single_task_id(monkeypatch):
    module = load_module()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--no-video"])

    args = module.parse_args()

    assert not hasattr(args, "task_id")


def test_run_episode_appends_success_to_video_filename(monkeypatch):
    module = load_module()
    written_video_paths = []

    class FakeEnv:
        def __init__(self, success):
            self.success = success
            self.unwrapped = self
            self.metadata = {"render_fps": 30}

        def reset(self, seed):
            return {}, {}

        def call(self, name):
            if name == "task_description":
                return ["pick up the object"]
            if name == "_max_episode_steps":
                return [1]
            raise AssertionError(f"unexpected env.call({name!r})")

        def step(self, action):
            return (
                {},
                np.zeros(1),
                np.ones(1, dtype=bool),
                np.zeros(1, dtype=bool),
                {"is_success": np.asarray([self.success])},
            )

    class FakeClient:
        def predict(self, images, state, prompt, reset):
            return np.zeros((10, 7)), {"total_ms": 1.0}

    monkeypatch.setattr(module, "preprocess_observation", lambda observation: observation)
    monkeypatch.setattr(module, "extract_model_inputs", lambda observation: ([object(), object()], object()))
    monkeypatch.setattr(module, "_render_one", lambda env: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "write_video", lambda path, frames, fps: written_video_paths.append(path))
    monkeypatch.setattr(Path, "mkdir", lambda self, parents, exist_ok: None)

    for success, suffix in ((True, "true"), (False, "false")):
        result = module.run_episode(
            FakeEnv(success),
            lambda observation: observation,
            FakeClient(),
            seed=1000,
            save_video_to=Path("videos/task_0_episode_0.mp4"),
        )

        expected = f"videos/task_0_episode_0_{suffix}.mp4"
        assert written_video_paths[-1] == expected
        assert result["video"] == expected


def test_run_episode_aggregates_paper_latency_components(monkeypatch):
    module = load_module()

    class FakeEnv:
        def __init__(self):
            self.steps = 0

        def reset(self, seed):
            return {}, {}

        def call(self, name):
            if name == "task_description":
                return ["pick up the object"]
            if name == "_max_episode_steps":
                return [11]
            raise AssertionError(f"unexpected env.call({name!r})")

        def step(self, action):
            self.steps += 1
            done = self.steps == 11
            return (
                {},
                np.zeros(1),
                np.asarray([done]),
                np.zeros(1, dtype=bool),
                {"is_success": np.asarray([done])},
            )

    class FakeClient:
        def __init__(self):
            self.metadata = iter((
                {"vit_ms": 1.0, "encode_ms": 2.0, "decode_ms": 3.0, "total_ms": 5.0},
                {"vit_ms": 4.0, "encode_ms": 5.0, "decode_ms": 6.0, "total_ms": 11.0},
            ))

        def predict(self, images, state, prompt, reset):
            return np.zeros((10, 7)), next(self.metadata)

    monkeypatch.setattr(module, "preprocess_observation", lambda observation: observation)
    monkeypatch.setattr(module, "extract_model_inputs", lambda observation: ([object(), object()], object()))

    result = module.run_episode(
        FakeEnv(),
        lambda observation: observation,
        FakeClient(),
        seed=1000,
        save_video_to=None,
    )

    assert result["inference_calls"] == 2
    assert result["total_inference_ms"] == 16.0
    assert result["total_vit_ms"] == 5.0
    assert result["total_encode_ms"] == 7.0
    assert result["total_decode_ms"] == 9.0
    assert result["full_model_inference_ms"] == 21.0
    assert result["average_model_inference_ms"] == 10.5


def test_main_evaluates_all_ten_tasks_and_aggregates_results(monkeypatch):
    module = load_module()
    seen_task_ids = []
    closed_task_ids = []
    written_reports = {}

    class FakeEnv:
        def __init__(self, task_id):
            self.task_id = task_id

        def close(self):
            closed_task_ids.append(self.task_id)

    def fake_make_env(config, n_envs, use_async_envs):
        assert n_envs == 1
        assert use_async_envs is False
        task_id = config.task_ids[0]
        seen_task_ids.append(task_id)
        return {"libero_object": {task_id: FakeEnv(task_id)}}

    def fake_run_episode(env, env_preprocessor, client, *, seed, save_video_to):
        return {
            "success": env.task_id % 2 == 0,
            "steps": 1,
            "inference_calls": 1,
            "total_inference_ms": 1.0,
            "task": f"task {env.task_id}",
            "seed": seed,
            "video": None,
        }

    args = SimpleNamespace(
        server_url="http://127.0.0.1:8080",
        client_path=Path("unused.py"),
        suite="libero_object",
        episodes=2,
        seed=1000,
        episode_length=None,
        output_dir=Path("unused-output"),
        no_video=True,
    )
    fake_client_module = SimpleNamespace(JetsonPiLiberoClient=lambda base_url: object())

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_client_module", lambda path: fake_client_module)
    monkeypatch.setattr(module, "make_env", fake_make_env)
    monkeypatch.setattr(module, "PolicyProcessorPipeline", lambda steps: object())
    monkeypatch.setattr(module, "LiberoProcessorStep", lambda: object())
    monkeypatch.setattr(module, "run_episode", fake_run_episode)
    monkeypatch.setattr(Path, "mkdir", lambda self, parents, exist_ok: None)
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, data, encoding: written_reports.setdefault(str(self), data),
    )

    module.main()

    assert seen_task_ids == list(range(10))
    assert closed_task_ids == list(range(10))

    report = json.loads(written_reports["unused-output/eval_info.json"])
    assert report["task_ids"] == list(range(10))
    assert report["episodes_per_task"] == 2
    assert report["total_episodes"] == 20
    assert report["success_rate"] == 0.5
    assert report["per_task_success_rate"] == {
        str(task_id): (1.0 if task_id % 2 == 0 else 0.0)
        for task_id in range(10)
    }
    assert len(report["results"]) == 20
    assert all("task_id" in result and "episode" in result for result in report["results"])
