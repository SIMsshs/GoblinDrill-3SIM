"""Beginner-friendly entry point for the Goblin Drill training pipeline.

Each subcommand deliberately performs one resumable stage. The expensive
stages are not chained automatically: inspect validation before spending GPU
time on the next stage.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs" / "decks" / "goblin_drill_evo_hero.json"
DEFAULT_DATA = ROOT / "data" / "rs_rows_drill"
BASE_CKPT = ROOT / "runs" / "tokens" / "drill_base_bc" / "best.pt"
TARGET_CKPT = ROOT / "runs" / "tokens" / "drill_target_bc" / "best.pt"


def _run(args: list[object]) -> int:
    command = [str(item) for item in args]
    print("\n> " + subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def _need(path: Path, explanation: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing: {path}\n{explanation}")


def doctor(args) -> int:
    from sim.deck_config import load_deck_config

    cfg = load_deck_config(args.deck_config)
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Deck:   {cfg.description}")
    print(f"Forms:  {list(cfg.forms)}  (0 base, 1 evolution, 2 hero)")
    missing = []
    for package in ("numpy", "pyarrow", "royalesim", "royaleimitate", "torch"):
        try:
            module = importlib.import_module(package)
            print(f"{package:14} OK  {getattr(module, '__version__', '')}")
        except Exception as exc:
            missing.append(package)
            print(f"{package:14} MISSING  {exc}")
    if not missing:
        import torch
        if torch.cuda.is_available():
            print(f"GPU:     {torch.cuda.get_device_name(0)}; CUDA {torch.version.cuda}")
        else:
            print("GPU:     CUDA not available (the pipeline works, but training will be slow)")
    free = shutil.disk_usage(ROOT).free / (1024 ** 3)
    corpus = ROOT / "data" / "il_replay"
    replays = len(list((corpus / "replays").glob("part-*.parquet")))
    actions = len(list((corpus / "actions").glob("part-*.parquet")))
    shards = len(list(args.data.glob("shard-*.npz")))
    print(f"Disk free: {free:.1f} GiB")
    print(f"Replay parts: replays={replays}, actions={actions}")
    print(f"Training shards: {shards} in {args.data}")
    if missing:
        print("\nInstall the missing packages; see docs/GOBLIN_DRILL_TRAINING_KO.md")
        return 1
    if sys.version_info[:2] != (3, 12):
        print("\nWARNING: this repository is tested with Python 3.12.")
    return 0


def data(args) -> int:
    command: list[object] = [sys.executable, ROOT / "tools" / "il_replay_download.py"]
    if args.parts:
        command += ["--parts", args.parts]
    return _run(command)


def rows(args) -> int:
    command: list[object] = [
        sys.executable, ROOT / "tools" / "rs_rows.py",
        "--deck-config", args.deck_config,
        "--out", args.data,
        "--others", args.others,
        "--workers", args.workers,
        "--batch", args.batch,
        "--seed", args.seed,
    ]
    if args.limit_parts:
        command += ["--limit-parts", args.limit_parts]
    if not args.full_sim_observation:
        command.append("--live-compat")
    if args.overwrite:
        command.append("--overwrite")
    return _run(command)


def pretrain(args) -> int:
    _need(args.data / "vocab.json", "Run `python tools/drill_pipeline.py rows` first.")
    return _run([
        sys.executable, "-m", "sim.train_tokens", "--name", args.name,
        "--data", args.data, "--epochs", args.epochs, "--batch", args.batch,
        "--lr", args.lr, "--wait-keep", args.wait_keep, "--beta", 0,
        "--ability-weight", args.ability_weight,
    ])


def finetune(args) -> int:
    _need(args.init, "Run the `pretrain` stage first, or pass --init explicitly.")
    return _run([
        sys.executable, "-m", "sim.train_tokens", "--name", args.name,
        "--data", args.data, "--init", args.init, "--finetune-target",
        "--epochs", args.epochs, "--batch", args.batch, "--lr", args.lr,
        "--wait-keep", args.wait_keep, "--phase-balance", args.phase_balance,
        "--ability-weight", args.ability_weight, "--beta", 0,
    ])


def evaluate(args) -> int:
    command: list[object] = [
        sys.executable, ROOT / "tools" / "rs_eval.py",
        "--deck-config", args.deck_config, "--matches", args.matches,
        "--workers", args.workers, "--delay", args.delay, "--lead", args.lead,
        "--rebuild",
    ]
    if args.pro:
        command.append("--pro")
    else:
        _need(args.ckpt, "Run the `finetune` stage first, or pass --ckpt explicitly.")
        command += ["--ckpt", args.ckpt]
    return _run(command)


def selfplay(args) -> int:
    _need(args.init, "Run the `finetune` stage first, or pass --init explicitly.")
    _need(args.opponent, "Run the `pretrain` stage first, or pass --opponent explicitly.")
    return _run([
        sys.executable, ROOT / "tools" / "rs_selfplay.py", "loop",
        "--name", args.name, "--deck-config", args.deck_config,
        "--init", args.init, "--ref", args.ref or args.init,
        "--opponent", args.opponent, "--eval-opponent", args.opponent,
        "--iters", args.iters, "--matches", args.matches,
        "--workers", args.workers, "--eval-every", args.eval_every,
        "--eval-matches", args.eval_matches, "--lr", args.lr, "--kl", args.kl,
    ])


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--deck-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("doctor", help="check Python, packages, GPU, data and deck")
    _common(p)
    p.set_defaults(func=doctor)

    p = sub.add_parser("data", help="download and verify the public replay dataset")
    p.add_argument("--parts", type=int, default=0, help="0 = complete dataset")
    p.set_defaults(func=data)

    p = sub.add_parser("rows", help="re-drive replays and build model-ready rows")
    _common(p)
    p.add_argument("--others", type=int, default=20000,
                   help="non-target matches used to learn general fundamentals")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-parts", type=int, default=0, help="0 = every downloaded part")
    p.add_argument("--full-sim-observation", action="store_true",
                   help="use hidden simulator features; not suitable for live transfer")
    p.add_argument("--overwrite", action="store_true",
                   help="explicitly replace existing generated shards in --data")
    p.set_defaults(func=rows)

    p = sub.add_parser("pretrain", help="plain behaviour cloning on all selected decks")
    _common(p)
    p.add_argument("--name", default="drill_base_bc")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wait-keep", type=float, default=0.35)
    p.add_argument("--ability-weight", type=float, default=3.0)
    p.set_defaults(func=pretrain)

    p = sub.add_parser("finetune", help="specialise on the exact target deck/forms")
    _common(p)
    p.add_argument("--init", type=Path, default=BASE_CKPT)
    p.add_argument("--name", default="drill_target_bc")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--wait-keep", type=float, default=0.35)
    p.add_argument("--phase-balance", type=float, default=0.25)
    p.add_argument("--ability-weight", type=float, default=4.0)
    p.set_defaults(func=finetune)

    p = sub.add_parser("eval", help="held-out evaluation; --pro gives the human baseline")
    p.add_argument("--deck-config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--ckpt", type=Path, default=TARGET_CKPT)
    p.add_argument("--pro", action="store_true")
    p.add_argument("--matches", type=int, default=100)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--delay", type=int, default=21)
    p.add_argument("--lead", type=int, default=-21)
    p.set_defaults(func=evaluate)

    p = sub.add_parser("selfplay", help="PPO fine-tuning with an imitation KL brake")
    p.add_argument("--deck-config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--init", type=Path, default=TARGET_CKPT)
    p.add_argument("--ref", type=Path, default=None)
    p.add_argument("--opponent", type=Path, default=BASE_CKPT)
    p.add_argument("--name", default="drill_selfplay")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--matches", type=int, default=48)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-matches", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl", type=float, default=0.05)
    p.set_defaults(func=selfplay)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
