#!/usr/bin/env python3
"""Batch runner for collect_lerobot_v4.py.

Runs single-part LeRobot v4 collectors in separate subprocesses.
This is intentional: collect_lerobot_v4.py starts and closes Isaac Sim, so a
batch script should not import it directly in-process.

Editable inline job format:
    JOBS = [
        {"parts": ["part_a", "part_b"], "random": "Yes", "runs": 3},
        {"parts": ["part_c"], "random": "No", "runs": 1},
    ]

Equivalent JSON file format accepted by --jobs-json:
    [
      {"parts": ["part_a", "part_b"], "random": true, "runs": 3},
      {"part": "part_c", "random": false}
    ]

In this script, random=True means collect_lerobot_v4.py is called with
--disturb. The expert action label remains the non-disturbed EE target; the
observation reflects the disturbed rollout state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


# Edit this list directly if you do not want to pass --jobs-json.
# Replace the example part names with actual names from param_config.PART_CONFIG.
JOBS: list[dict[str, Any]] = [
    {"parts": ["usb_a"], "random": "Yes", "runs": 20},
    {"parts": ["usb_a"], "random": "No", "runs": 30},
]

@dataclass(frozen=True)
class BatchItem:
    part_name: str
    disturb: bool
    run_index: int
    seed: int | None
    output_root: Path
    repo_id: str
    label_json: Path
    results_json: Path
    summary_json: Path


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "t", "1", "random", "disturb"}:
        return True
    if text in {"no", "n", "false", "f", "0", "none", "not", "clean"}:
        return False
    raise ValueError(f"Cannot parse boolean/random value: {value!r}")


def _slug(text: str) -> str:
    text = text.strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "part"


def _load_jobs(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        jobs = JOBS
    else:
        jobs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(jobs, list):
        raise ValueError("Jobs must be a list of objects")
    return jobs


def _iter_parts(job: dict[str, Any]) -> Iterable[str]:
    if "parts" in job:
        parts = job["parts"]
    elif "part" in job:
        parts = [job["part"]]
    else:
        raise ValueError(f"Job missing 'parts' or 'part': {job!r}")

    if isinstance(parts, str):
        # Allow comma-separated strings, but preserve a plain single name.
        if "," in parts:
            parts = [p.strip() for p in parts.split(",") if p.strip()]
        else:
            parts = [parts]
    if not isinstance(parts, list) or not parts:
        raise ValueError(f"Invalid parts field: {parts!r}")
    for part in parts:
        part_name = str(part).strip()
        if not part_name:
            raise ValueError(f"Empty part name in job: {job!r}")
        yield part_name


def _expand_jobs(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> list[BatchItem]:
    items: list[BatchItem] = []
    for job_idx, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"Each job must be an object, got: {job!r}")

        disturb = _parse_bool(job.get("random", job.get("disturb", False)))
        runs = int(job.get("runs", job.get("repeat", args.runs_per_item)))
        if runs <= 0:
            raise ValueError(f"runs must be > 0, got {runs} for job {job!r}")

        for part_name in _iter_parts(job):
            part_slug = _slug(part_name)
            mode = "disturb" if disturb else "clean"
            for run_idx in range(runs):
                global_idx = len(items)
                seed = None
                if disturb:
                    seed = int(args.seed_base + global_idx)

                run_name = f"{part_slug}_{mode}_{run_idx:03d}"
                output_root = args.output_root / run_name
                label_json = output_root.parent / f"{output_root.name}_label.json"
                results_json = output_root.parent / f"{output_root.name}_results.json"
                summary_json = output_root / "taskboard_lerobot_summary.json"
                repo_id = f"{args.repo_prefix}/{run_name}"

                items.append(BatchItem(
                    part_name=part_name,
                    disturb=disturb,
                    run_index=run_idx,
                    seed=seed,
                    output_root=output_root,
                    repo_id=repo_id,
                    label_json=label_json,
                    results_json=results_json,
                    summary_json=summary_json,
                ))
    return items


def _build_command(args: argparse.Namespace, item: BatchItem) -> list[str]:
    cmd = [
        str(args.python),
        str(args.collect_script),
        "--part-name", item.part_name,
        "--output-root", str(item.output_root),
        "--repo-id", item.repo_id,
        "--label-json", str(item.label_json),
        "--results-json", str(item.results_json),
        "--sample-hz", str(args.sample_hz),
        "--max-recorded-frames", str(args.max_recorded_frames),
        "--max-sim-steps", str(args.max_sim_steps),
    ]

    if args.overwrite:
        cmd.append("--overwrite")

    if args.include_depth:
        cmd.append("--include-depth")
    else:
        cmd.append("--no-include-depth")

    if args.skip_non_ee_actions:
        cmd.append("--skip-non-ee-actions")
    else:
        cmd.append("--no-skip-non-ee-actions")

    if item.disturb:
        cmd.extend([
            "--disturb",
            "--disturbance-prob", str(args.disturbance_prob),
            "--disturbance-duration-steps", str(args.disturbance_duration_steps),
            "--disturbance-pos-max", str(args.disturbance_pos_max),
            "--disturbance-rot-max-deg", str(args.disturbance_rot_max_deg),
        ])
        if item.seed is not None:
            cmd.extend(["--disturbance-seed", str(item.seed)])
    else:
        cmd.append("--no-disturb")

    if args.extra_v4_args:
        cmd.extend(args.extra_v4_args)
    return cmd


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc), "path": str(path)}


def _write_batch_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loop over part/random jobs and call collect_lerobot_v4.py once per single-part rollout."
    )
    parser.add_argument(
        "--jobs-json",
        type=Path,
        default=None,
        help="Optional JSON list of jobs. If omitted, the editable JOBS list in this file is used.",
    )
    parser.add_argument(
        "--collect-script",
        type=Path,
        default=Path(__file__).with_name("collect_lerobot_v4.py"),
        help="Path to collect_lerobot_v4.py.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to run collect_lerobot_v4.py. Use the Isaac Sim Python env.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/lerobot_v4_batch"))
    parser.add_argument("--repo-prefix", type=str, default="taskboard/v4_batch")
    parser.add_argument("--batch-summary-json", type=Path, default=None)

    parser.add_argument("--runs-per-item", type=int, default=1, help="Default runs for a job when it has no runs field.")
    parser.add_argument("--sample-hz", type=float, default=30.0)
    parser.add_argument("--max-recorded-frames", type=int, default=0)
    parser.add_argument("--max-sim-steps", type=int, default=0)
    parser.add_argument("--include-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-non-ee-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--disturbance-prob", type=float, default=0.004)
    parser.add_argument("--disturbance-duration-steps", type=int, default=20)
    parser.add_argument("--disturbance-pos-max", type=float, default=0.005)
    parser.add_argument("--disturbance-rot-max-deg", type=float, default=2.0)

    parser.add_argument(
        "--stop-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the batch on the first failed subprocess.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "extra_v4_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are appended to every collect_lerobot_v4.py call.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.collect_script = args.collect_script.resolve()
    args.output_root = args.output_root.resolve()
    if args.batch_summary_json is None:
        args.batch_summary_json = args.output_root / "batch_summary.json"
    else:
        args.batch_summary_json = args.batch_summary_json.resolve()

    if not args.collect_script.exists():
        raise FileNotFoundError(f"collect script not found: {args.collect_script}")
    if args.runs_per_item <= 0:
        raise ValueError("--runs-per-item must be > 0")

    jobs = _load_jobs(args.jobs_json)
    items = _expand_jobs(args, jobs)
    if not items:
        raise ValueError("No jobs to run. Fill JOBS or pass --jobs-json.")

    args.output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("ISAACSIM_HEADLESS", "1")

    batch_payload: dict[str, Any] = {
        "created_at_unix": time.time(),
        "collect_script": str(args.collect_script),
        "python": str(args.python),
        "output_root": str(args.output_root),
        "repo_prefix": args.repo_prefix,
        "total_runs": len(items),
        "settings": {
            "sample_hz": args.sample_hz,
            "max_recorded_frames": args.max_recorded_frames,
            "max_sim_steps": args.max_sim_steps,
            "include_depth": args.include_depth,
            "skip_non_ee_actions": args.skip_non_ee_actions,
            "disturbance_prob": args.disturbance_prob,
            "disturbance_duration_steps": args.disturbance_duration_steps,
            "disturbance_pos_max": args.disturbance_pos_max,
            "disturbance_rot_max_deg": args.disturbance_rot_max_deg,
            "seed_base": args.seed_base,
        },
        "runs": [],
    }

    for i, item in enumerate(items, start=1):
        cmd = _build_command(args, item)
        print(f"[batch] {i}/{len(items)} part={item.part_name} disturb={item.disturb} out={item.output_root}", flush=True)
        print("[batch] command:", " ".join(cmd), flush=True)

        run_record: dict[str, Any] = {
            **asdict(item),
            "output_root": str(item.output_root),
            "label_json": str(item.label_json),
            "results_json": str(item.results_json),
            "summary_json": str(item.summary_json),
            "command": cmd,
        }

        if args.dry_run:
            run_record["returncode"] = None
            run_record["status"] = "dry_run"
            batch_payload["runs"].append(run_record)
            _write_batch_summary(args.batch_summary_json, batch_payload)
            continue

        start = time.time()
        proc = subprocess.run(cmd, env=env)
        elapsed = time.time() - start
        run_record["returncode"] = proc.returncode
        run_record["elapsed_s"] = elapsed
        run_record["status"] = "ok" if proc.returncode == 0 else "failed"
        run_record["label"] = _read_json_if_exists(item.label_json)
        run_record["summary"] = _read_json_if_exists(item.summary_json)
        run_record["results"] = _read_json_if_exists(item.results_json)
        batch_payload["runs"].append(run_record)
        _write_batch_summary(args.batch_summary_json, batch_payload)

        if proc.returncode != 0 and args.stop_on_error:
            print(f"[batch] stopping after failed run returncode={proc.returncode}", flush=True)
            return proc.returncode

    ok = sum(1 for r in batch_payload["runs"] if r.get("status") == "ok")
    failed = sum(1 for r in batch_payload["runs"] if r.get("status") == "failed")
    print(f"[batch] done. ok={ok}, failed={failed}, summary={args.batch_summary_json}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
