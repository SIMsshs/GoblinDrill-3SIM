"""Train `sim.model_tokens.TokenPolicy` on the shards from tools/build_rows.py.

    .venvs/buildabot/Scripts/python.exe -m sim.train_tokens --name bc --beta 0
    .venvs/buildabot/Scripts/python.exe -m sim.train_tokens --name awr --beta 0.1
    ... --finetune-target --init runs/tokens/base_bc/best.pt --name drill_bc

Only numpy and torch are needed, so it runs in any environment with CUDA
PyTorch; it does not import the engine.

Rows: by default only *trusted* rows (before the match's first divergence)
are used, and wait rows are subsampled to `--wait-keep` so the dataset fits
in memory. Validation is the hash split written by the builder, by match.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .model_tokens import TokenPolicy, losses

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "il_rows"
RUNS = ROOT / "runs" / "tokens"
COLS = ("unit_ids", "unit_feats", "towers", "scalars", "hand", "next", "opp_seen",
        "opp_last", "opp_last_age", "ability_ids", "ability_feats", "outcome", "won",
        "act_card", "act_x", "act_y", "act_gate")


def _legacy_gate(hand: np.ndarray, card: np.ndarray) -> np.ndarray:
    out = np.full(len(card), -1, dtype=np.int16)
    out[card == 0] = 0
    for slot in range(4):
        hit = (card > 0) & (hand[:, slot] == card) & (out < 0)
        out[hit] = slot + 1
    return out


def load(split: str, wait_keep: float, trusted_only: bool, target_only: bool, seed: int = 0,
         root: Path = DATA):
    rng = np.random.default_rng(seed)
    parts = {c: [] for c in COLS}
    for path in sorted(root.glob("shard-*.npz")):
        z = np.load(path)
        keep = z["split"] == split
        if trusted_only:
            keep &= z["trusted"]
        if target_only:
            selector = "target" if "target" in z.files else "hog"
            if selector not in z.files:
                raise KeyError(f"{path}: no target (or legacy hog) column; rebuild rows")
            keep &= z[selector]
        waits = z["act_card"] == 0
        keep &= ~waits | (rng.random(len(waits)) < wait_keep)
        n = len(z["act_card"])
        for c in COLS:
            if c in z.files:
                value = z[c]
            elif c == "ability_ids":
                value = np.zeros((n, 3), dtype=np.int16)
            elif c == "ability_feats":
                value = np.zeros((n, 3, 4), dtype=np.float16)
            elif c == "act_gate":
                value = _legacy_gate(z["hand"], z["act_card"])
            else:
                raise KeyError(f"{path}: missing required column {c}")
            parts[c].append(value[keep])
    if not parts["act_card"]:
        raise FileNotFoundError(f"no shard-*.npz files under {root}")
    data = {c: np.concatenate(v) for c, v in parts.items()}
    if not len(data["act_card"]):
        scope = "target deck " if target_only else ""
        raise ValueError(f"no {scope}{split} rows in {root}; build more replay parts")
    data["unit_feats"] = data["unit_feats"].astype(np.float16)
    return data


PHASE_NAMES = [f"{t}/{w}" for t in ("normal", "double", "overtime")
               for w in ("all up", "theirs down", "ours down", "both down")]


def phase_of(data):
    """Phase bucket per row: clock (normal / double elixir / overtime) x towers
    (all princess towers up / one of theirs down / one of ours down / both).

    `scalars` = [elixir, clock, double elixir flag, overtime flag];
    `towers` = [own L, own R, own king, enemy L, enemy R, enemy king] as hp
    fractions, in the acting player's frame.
    """
    sc, tw = data["scalars"], data["towers"]
    clock = np.where(sc[:, 3] > 0.5, 2, np.where(sc[:, 2] > 0.5, 1, 0))
    theirs = (tw[:, 3] <= 0) | (tw[:, 4] <= 0)
    ours = (tw[:, 0] <= 0) | (tw[:, 1] <= 0)
    return clock * 4 + theirs.astype(int) + 2 * ours.astype(int)


def balanced_order(data, alpha: float, rng):
    """One epoch of row indices, drawn so rare phases weigh more.

    Each row's weight is (1 / rows in its phase) ** alpha: alpha 0 is the
    data as it is, 1 is every phase equally often. Repeats of a tiny phase
    are capped at 20x its size per epoch, so a handful of rows is not drilled
    thousands of times.
    """
    ph = phase_of(data)
    counts = np.bincount(ph, minlength=12).astype(float)
    w = (1.0 / np.maximum(counts[ph], 1)) ** alpha
    n = len(ph)
    share = w / w.sum()
    # cap: expected draws of a phase <= 20 x its rows
    per_phase = np.bincount(ph, weights=share, minlength=12) * n
    scale = np.where(per_phase > 20 * counts, 20 * counts / np.maximum(per_phase, 1e-9), 1.0)
    share = share * scale[ph]
    share /= share.sum()
    return rng.choice(n, n, replace=True, p=share)


def batches(data, size: int, shuffle: bool, device, order=None):
    n = len(data["act_card"])
    if order is None:
        order = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, size):
        idx = order[i:i + size]
        b = {}
        for c, v in data.items():
            t = torch.from_numpy(np.ascontiguousarray(v[idx]))
            if t.dtype in (torch.int16, torch.int32):
                t = t.long()
            if t.dtype == torch.float64:
                t = t.float()
            b[c] = t.to(device, non_blocking=True)
        yield b


@torch.no_grad()
def evaluate(model, data, device, beta, ability_weight=1.0):
    model.eval()
    tot, n = {}, 0
    for b in batches(data, 2048, False, device):
        _, s = losses(model, b, beta=beta, ability_weight=ability_weight)
        k = len(b["act_card"])
        for key, v in s.items():
            tot[key] = tot.get(key, 0.0) + v * k
        n += k
    model.train()
    return {k: v / max(1, n) for k, v in tot.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="0 = plain imitation; >0 = outcome-weighted (AWR temperature)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wait-keep", type=float, default=0.35)
    ap.add_argument("--all-rows", action="store_true",
                    help="include rows after the match's divergence point")
    ap.add_argument("--finetune-target", action="store_true",
                    help="train only rows whose deck (including configured forms) is the target")
    ap.add_argument("--finetune-hog", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--init", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--phase-balance", type=float, default=0.0,
                    help="0 = rows as they come; >0 = draw rare game phases (overtime, "
                         "a tower down) more often, weight (1/phase rows)**this")
    ap.add_argument("--ability-weight", type=float, default=3.0,
                    help="extra gate-loss weight for rare hero/champion ability presses")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((args.data / "vocab.json").read_text())
    t0 = time.time()
    target_only = args.finetune_target or args.finetune_hog
    train = load("train", args.wait_keep, not args.all_rows, target_only, args.seed, args.data)
    val = load("val", args.wait_keep, True, target_only, 1, args.data)
    val_target = load("val", args.wait_keep, True, True, 1, args.data)
    print(f"rows: train {len(train['act_card']):,}  val {len(val['act_card']):,}  "
          f"val(target) {len(val_target['act_card']):,}  ({time.time() - t0:.0f}s)", flush=True)

    meta = dict(meta, d=args.d, layers=args.layers, heads=args.heads)
    model = TokenPolicy(meta["n_cards"], meta["n_units"], d=args.d, layers=args.layers,
                        heads=args.heads, ability_actions=meta.get("ability_actions", 0),
                        ability_features=meta.get("ability_features", 4)).to(device)
    if args.init:
        init = torch.load(args.init, map_location=device)
        missing, unexpected = model.load_state_dict(init["model"], strict=False)
        if unexpected:
            raise ValueError(f"checkpoint has unexpected parameters: {unexpected}")
        if missing:
            allowed = ("ability_in.", "ability_slot_emb.", "ability_type", "ability_query.")
            if any(not key.startswith(allowed) for key in missing):
                raise ValueError(f"checkpoint is not architecture-compatible; missing {missing}")
            print("init checkpoint has no ability head; initialized the new ability head randomly", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train["act_card"]) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}))
    best = -1.0
    log = (out / "log.jsonl").open("a")
    step = 0
    rng = np.random.default_rng(args.seed)
    if args.phase_balance > 0:
        counts = np.bincount(phase_of(train), minlength=12)
        print("train rows per phase: " + ", ".join(f"{PHASE_NAMES[i]} {c:,}"
                                                   for i, c in enumerate(counts) if c), flush=True)
    for epoch in range(args.epochs):
        order = balanced_order(train, args.phase_balance, rng) if args.phase_balance > 0 else None
        for b in batches(train, args.batch, True, device, order=order):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, s = losses(model, b, beta=args.beta, ability_weight=args.ability_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                print(f"  ep {epoch} step {step}/{steps} loss {loss.item():.3f} "
                      f"gate {s['gate_acc']:.3f} card {s['card_acc']:.3f} "
                      f"ability {s['ability_acc']:.3f} tile {s['tile_acc']:.3f} "
                      f"near {s['tile_near']:.3f} w {s['mean_w']:.2f}",
                      flush=True)
        v = evaluate(model, val, device, 0.0)
        vt = evaluate(model, val_target, device, 0.0)
        # Per phase, so a weak phase is not hidden in the average.
        ph = phase_of(val_target)
        per_phase = {}
        for i in range(12):
            m = ph == i
            if m.sum() >= 200:
                sub = {c: arr[m] for c, arr in val_target.items()}
                e = evaluate(model, sub, device, 0.0)
                per_phase[PHASE_NAMES[i]] = {"rows": int(m.sum()), "card": round(e["card_acc"], 3),
                                             "near": round(e["tile_near"], 3)}
        print("  target per phase: " + "; ".join(f"{k} ({v_['rows']}) card {v_['card']} near {v_['near']}"
                                                 for k, v_ in per_phase.items()), flush=True)
        score = vt["card_acc"] + vt["tile_near"] + 0.25 * vt["ability_acc"]
        rec = {"epoch": epoch, "val": v, "val_target": vt, "val_target_phase": per_phase,
               "time": time.time() - t0}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"epoch {epoch}: val card {v['card_acc']:.3f} tile {v['tile_acc']:.3f} "
              f"near {v['tile_near']:.3f} gate {v['gate_acc']:.3f} | target card {vt['card_acc']:.3f} "
              f"near {vt['tile_near']:.3f} ability {vt['ability_acc']:.3f}", flush=True)
        torch.save({"model": model.state_dict(), "meta": meta, "epoch": epoch}, out / "last.pt")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "meta": meta, "epoch": epoch}, out / "best.pt")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
