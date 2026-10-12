"""Self-play reinforcement learning in RoyaleSim, starting from imitation.

Everything up to now copied pros. This lets the target-deck model learn from its own
results: it plays full RoyaleSim battles under the live configuration (21-tick
command delay, decisions on a board rebuilt from public information and run
21 ticks ahead) against a frozen reactive opponent playing the real
opponents' decks from IL_Replay's train split, and is pushed toward the
decisions of the battles it won.

Two halves, because the engine and the GPU live in different environments:

  collect  (.venvs/royalegym, CPU, engine)  - N battles -> one .npz of decisions
  train    (.venvs/buildabot, CUDA)         - one PPO update on that .npz

and a driver that alternates them and evaluates every few iterations:

  .venvs\\royalegym\\Scripts\\python.exe tools\\rs_selfplay.py loop --iters 40

The policy being trained is exactly the one the live bot runs: the gate's
wait / play probabilities with the evaluator's wait correction and its
p_wait ** (DECIDE / 20) cadence, a slot among the affordable cards, and a tile
among the legal ones. Here the slot and tile are SAMPLED (the evaluator takes
the argmax), so the policy can find out what else works; that is what the
log-probabilities below describe.

Update: PPO uses an objective-aligned terminal return: win/draw/loss remains
the dominant signal, with small crown and remaining-tower terms so an initial
policy that loses every hard match can still distinguish a close loss from a
three-crown collapse.  A curriculum mixes recorded ghosts with progressively
faster reactive opponents, while a KL penalty keeps the policy near imitation.

Nothing here touches the emulator or the game.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

RUNS = ROOT / "runs" / "selfplay"
KEYS = ("unit_ids", "unit_feats", "towers", "scalars", "hand", "opp_seen", "opp_last",
        "opp_last_age", "ability_ids", "ability_feats")
DECIDE = 10
TOP_TILES = 8          # tiles sampled among the model's best legal ones
EXTRA_TILES = 8        # plus this many random legal tiles from anywhere
EXPLORE = 0.15         # share of the tile choice spread evenly over all candidates
TILE_TEMP = 1.0
# Why the extra tiles: sampling only the model's 8 best legal tiles never once
# tried a pocket Hog after a princess tower fell - the bridge row alone holds
# 16 legal tiles, so all 8 were bridge tiles - and what is never tried cannot
# be learned (2026-10-10).
CANDS = TOP_TILES + EXTRA_TILES


def linear_schedule(start: float, end: float, iteration: int, steps: int) -> float:
    """Inclusive linear curriculum: iteration 0 is start, steps-1 is end."""
    if steps <= 1:
        return float(end)
    progress = min(1.0, max(0.0, iteration / (steps - 1)))
    return float(start + (end - start) * progress)


def shaped_return(result: str, crowns, tower_fraction, terminal_weight: float = 0.7,
                  crown_weight: float = 0.1, tower_weight: float = 0.2):
    """Return a bounded, game-objective reward and its auditable components.

    Crown and tower terms are deliberately subordinate to the terminal result:
    with the defaults, any win still outranks any draw and any draw outranks any
    loss.  They only rank outcomes inside those broad result classes.
    """
    weights = np.asarray([terminal_weight, crown_weight, tower_weight], dtype=float)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("reward weights must be non-negative with a positive sum")
    if terminal_weight <= 2 * (crown_weight + tower_weight):
        raise ValueError(
            "terminal weight must exceed twice the crown and tower weights combined "
            "so every win outranks every draw and every draw outranks every loss")
    weights /= weights.sum()
    terminal = {"win": 1.0, "draw": 0.5, "loss": 0.0}[result]
    crown_diff = float(crowns[0]) - float(crowns[1])
    tower_diff = float(tower_fraction[0]) - float(tower_fraction[1])
    crown_score = float(np.clip(0.5 + crown_diff / 6.0, 0.0, 1.0))
    tower_score = float(np.clip(0.5 + tower_diff / 6.0, 0.0, 1.0))
    reward = float(weights @ np.asarray([terminal, crown_score, tower_score]))
    return reward, {"terminal": terminal, "crown": crown_score, "tower": tower_score,
                    "crown_diff": crown_diff, "tower_diff": tower_diff}


def evaluation_score(summary):
    """Lexicographic checkpoint rank: game result first, crown margin second."""
    n = max(1, int(summary.get("n", 0)))
    result = (float(summary.get("wins", 0)) + 0.5 * float(summary.get("draws", 0))) / n
    return result, float(summary.get("crown_diff", -999.0))


# ----------------------------------------------------------------------------
# collect (royalegym env)
# ----------------------------------------------------------------------------

def _sampling_agent():
    import rs_eval
    import rs_rows as RR

    class SPAgent(rs_eval.RSAgent):
        """RSAgent.act with a sampled slot and tile, recording every decision."""

        def __init__(self, ckpt, seed=0, wait_keep=0.35):
            super().__init__(ckpt, seed=seed, wait_keep=wait_keep)
            self.records = []

        def act(self, battle, state, team, mem, tick, sub, royalesim):
            torch = self.torch
            self._team = team
            view = self.ahead(battle, state)
            row = RR.obs_row(view, team, sub, mem, tick + self.lead, live=self.live)
            b = self.batch(row)
            with torch.no_grad():
                gate, _, value = self.model(b, card_ids=b["hand"][:, 0].clamp(min=0))
            g = gate[0].float().numpy().astype(np.float64)
            g[0] += self.wait_bias
            p = state["players"][team]
            self.mask_gate(g, battle, state, team)
            mask = g > -1e8
            mask[0] = True
            if not mask[1:].any():
                return None
            q = gate_probs(g, mask)
            a = int(self.rng.choice(len(q), p=q))
            rec = {k: np.asarray(row[k]) for k in KEYS}
            rec.update(next=int(row["next"]), mask=mask, gate_a=a, logq_gate=float(math.log(q[a])),
                       v=float(1 / (1 + math.exp(-float(value[0, 1])))), tile_a=-1, logq_tile=0.0,
                       tile_cands=np.full(CANDS, -1, np.int16), tick=tick)
            if a == 0:
                self.records.append(rec)
                return None
            if a >= 5:
                self.records.append(rec)
                return (team, a - 1, 0, 0)
            slot = a - 1
            with torch.no_grad():
                _, tile, _ = self.model(b, card_ids=b["hand"][:, slot])
            logits = tile[0].float().numpy().astype(np.float64)

            def legal_at(t):
                tx, ty = int(t % 18), int(t // 18)
                wx, wy = rs_eval.to_world(tx * 1000 + 500, ty * 1000 + 500, team)
                ok = battle.check_deploy(team, slot, wx * sub, wy * sub) == 0
                return ok, wx, wy

            legal, seen = [], set()
            for t in np.argsort(-logits)[:60]:
                ok, wx, wy = legal_at(t)
                seen.add(int(t))
                if ok:
                    legal.append((int(t), wx, wy))
                    if len(legal) == TOP_TILES:
                        break
            if not legal:
                return None
            for t in self.rng.permutation(len(logits))[:120]:
                if len(legal) == CANDS:
                    break
                if int(t) in seen:
                    continue
                seen.add(int(t))
                ok, wx, wy = legal_at(t)
                if ok:
                    legal.append((int(t), wx, wy))
            pt = tile_probs(np.array([logits[t] for t, _, _ in legal]))
            k = int(self.rng.choice(len(legal), p=pt))
            t, wx, wy = legal[k]
            cands = np.full(CANDS, -1, np.int16)
            cands[:len(legal)] = [c for c, _, _ in legal]
            rec.update(tile_a=k, logq_tile=float(math.log(pt[k])), tile_cands=cands)
            self.records.append(rec)
            return (team, slot, wx * sub, wy * sub)

    return SPAgent


def tile_probs(lg):
    """Tile choice among the candidates: (1 - EXPLORE) x the model's softmax
    + EXPLORE spread evenly, so every candidate keeps a real chance."""
    p = np.exp((lg - lg.max()) / TILE_TEMP)
    p /= p.sum()
    return (1 - EXPLORE) * p + EXPLORE / len(lg)


def gate_probs(g, mask):
    """The live gate as a distribution over wait, four cards and abilities.

    Evaluator: wait with probability p_wait ** (DECIDE / 20), where p_wait is
    the softmax wait share; otherwise play. Here the slot played is drawn in
    proportion to the slot softmax, among affordable slots only.
    """
    g = np.where(mask, g, -1e9)
    pr = np.exp(g - g.max())
    pr /= pr.sum()
    qw = float(pr[0]) ** (DECIDE / 20)
    s = pr[1:].sum()
    q = np.zeros(len(g))
    q[0] = qw
    q[1:] = (1 - qw) * pr[1:] / max(s, 1e-12)
    q = np.where(mask, q, 0.0)
    return q / q.sum()


def _collect_one(task):
    import rs_eval
    rs_eval.AGENT_CLASS = _sampling_agent()
    out = rs_eval.play(task)
    if not out or not out.get("records"):
        return None
    weights = task[13:16] if len(task) >= 16 else (0.7, 0.1, 0.2)
    reward, components = shaped_return(out["result"], out["crowns"],
                                       out["tower_fraction"], *weights)
    source = json.loads(task[0]).get("source", {})
    kind = ("reactive_general" if source.get("synthetic_opponent") else
            "reactive_replay" if task[4] else "ghost")
    return {"outcome": out["result"], "crowns": out["crowns"], "reward": reward,
            "components": components, "records": out["records"], "kind": kind}


def pick_matches(n, seed, split="train", deck_config=None, general_opponents=0):
    """Target-deck templates, optionally given supported general opponent decks."""
    import random
    import pyarrow.parquet as pq
    import rs_eval
    import rs_rows as RR
    from sim.deck_config import load_deck_config
    from royalesim_redrive import CORPUS
    _, names, _, _ = RR._setup()
    target = load_deck_config(deck_config)
    parts = sorted((CORPUS / "replays").glob("part-*.parquet"))
    rnd = random.Random(seed)
    rnd.shuffle(parts)
    picked = []
    opponent_decks = []
    seen_decks = set()
    for part in parts:
        rows = pq.read_table(part, columns=["replay_tag", "game_mode", "payload_json"]).to_pylist()
        rnd.shuffle(rows)
        for r in rows:
            if r["game_mode"] not in RR.MODES or RR.split_of(r["replay_tag"]) != split:
                continue
            p = json.loads(r["payload_json"])
            battle = p.get("battle", {})
            if rs_eval.unsupported_deck_cards(battle, names):
                continue
            target_seat = target.target_seat(battle)
            if target_seat is not None and len(picked) < n:
                p.setdefault("source", {})["replay_tag"] = r["replay_tag"]
                picked.append(p)
            elif len(opponent_decks) < general_opponents:
                sides = ["team", "opponent"]
                rnd.shuffle(sides)
                for side in sides:
                    deck = battle[side]["players"][0]["deck"]
                    signature = tuple(sorted(str(card.get("card_key") or "") for card in deck))
                    if (len(deck) == 8 and signature not in seen_decks
                            and not target.matches_replay_cards(deck)):
                        seen_decks.add(signature)
                        opponent_decks.append(json.loads(json.dumps(deck)))
                        break
            if len(picked) >= n and len(opponent_decks) >= general_opponents:
                break
        if len(picked) >= n and len(opponent_decks) >= general_opponents:
            break

    replace_n = min(general_opponents, len(picked), len(opponent_decks))
    for index, deck in zip(rnd.sample(range(len(picked)), replace_n), opponent_decks[:replace_n]):
        p = picked[index]
        seat = target.target_seat(p["battle"])
        other = "opponent" if seat == 0 else "team"
        p["battle"][other]["players"][0]["deck"] = deck
        p.setdefault("source", {})["synthetic_opponent"] = True
    return [json.dumps(p) for p in picked]


def collect(args):
    from concurrent.futures import ProcessPoolExecutor
    import random
    # Validate once in the parent process so a bad custom reward configuration
    # produces a direct explanation instead of a remote worker traceback.
    shaped_return("draw", [0, 0], [3.0, 3.0], args.terminal_weight,
                  args.crown_weight, args.tower_weight)
    reactive_rate = min(1.0, max(0.0, float(args.reactive_rate)))
    desired_reactive = int(round(args.matches * reactive_rate))
    desired_general = int(round(desired_reactive * min(1.0, max(0.0, args.general_opponent_rate))))
    picked = pick_matches(args.matches, args.seed, deck_config=args.deck_config,
                          general_opponents=desired_general)
    synthetic = [bool(json.loads(text).get("source", {}).get("synthetic_opponent"))
                 for text in picked]
    reactive = {i for i, value in enumerate(synthetic) if value}
    candidates = [i for i in range(len(picked)) if i not in reactive]
    rng = random.Random(args.seed + 9173)
    rng.shuffle(candidates)
    reactive.update(candidates[:max(0, min(len(picked), desired_reactive) - len(reactive))])
    tasks = []
    for k, text in enumerate(picked):
        is_reactive = k in reactive
        tasks.append((text, args.ckpt, "agent", args.seed * 1000 + k,
                      args.opponent if is_reactive else "", 21, -21,
                      False, False, False, str(args.deck_config),
                      bool(args.randomize_reactive and is_reactive), args.opponent_decide,
                      args.terminal_weight, args.crown_weight, args.tower_weight))
    t0 = time.time()
    recs, results = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for res in pool.map(_collect_one, tasks):
            if res is None:
                continue
            results.append(res)
            for row in res["records"]:
                row["ret"] = res["reward"]
                recs.append(row)
    if not recs:
        raise RuntimeError(
            "self-play produced no decisions; download the complete replay corpus, "
            "check the exact deck/forms in --deck-config, and verify the checkpoint")
    out = {k: np.stack([x[k] for x in recs]) for k in recs[0]}
    np.savez_compressed(args.out, **out)
    def group(kind):
        return [r for r in results if r["kind"] == kind]

    def rate(rows):
        return sum(r["outcome"] == "win" for r in rows) / max(1, len(rows))

    rewards = np.asarray([r["reward"] for r in results], dtype=float)
    row = {"matches": len(results), "decisions": len(recs),
           "win_rate": round(rate(results), 3),
           "mean_reward": round(float(rewards.mean()), 4),
           "reward_std": round(float(rewards.std()), 4),
           "mean_crown_diff": round(float(np.mean([r["components"]["crown_diff"]
                                                    for r in results])), 3),
           "mean_tower_diff": round(float(np.mean([r["components"]["tower_diff"]
                                                    for r in results])), 3),
           "seconds": round(time.time() - t0)}
    for kind in ("ghost", "reactive_replay", "reactive_general"):
        rows = group(kind)
        row[f"{kind}_matches"] = len(rows)
        row[f"{kind}_win_rate"] = round(rate(rows), 3) if rows else None
    print(json.dumps(row), flush=True)


# ----------------------------------------------------------------------------
# train (buildabot env, CUDA)
# ----------------------------------------------------------------------------

def train(args):
    import torch
    import torch.nn.functional as F
    from sim.model_tokens import TokenPolicy
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def load(path):
        blob = torch.load(path, map_location="cpu")
        m = blob["meta"]
        net = TokenPolicy(m["n_cards"], m["n_units"], d=m.get("d", 192),
                          layers=m.get("layers", 4), heads=m.get("heads", 6),
                          ability_actions=m.get("ability_actions", 0),
                          ability_features=m.get("ability_features", 4))
        net.load_state_dict(blob["model"])
        return net.to(dev), blob

    model, blob = load(args.ckpt)
    # No dropout: the probabilities trained on must be the ones the agent
    # sampled from (with dropout on, ratios were off by 4% at learning rate 0).
    model.eval()
    ref, _ = load(args.ref)
    ref.eval()
    wait_bias = math.log(1 / float(args.wait_keep))
    d = np.load(args.data)
    n = len(d["gate_a"])
    adv_all = d["ret"] - d["v"]
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-6)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    stats = []
    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            b = {k: torch.as_tensor(d[k][idx]).to(dev) for k in KEYS}
            for k in ("unit_ids", "hand", "opp_seen", "opp_last", "ability_ids"):
                b[k] = b[k].long()
            b["unit_feats"] = b["unit_feats"].float()
            b["ability_feats"] = b["ability_feats"].float()
            b["opp_last_age"] = b["opp_last_age"].float()
            b["next"] = torch.as_tensor(d["next"][idx]).long().to(dev)
            mask = torch.as_tensor(d["mask"][idx]).to(dev)
            a = torch.as_tensor(d["gate_a"][idx]).long().to(dev)
            old = torch.as_tensor(d["logq_gate"][idx] + d["logq_tile"][idx]).float().to(dev)
            adv = torch.as_tensor(adv_all[idx]).float().to(dev)
            ret = torch.as_tensor(d["ret"][idx]).float().to(dev)
            hand = b["hand"]
            played = (a >= 1) & (a <= 4)
            slot = (a - 1).clamp(min=0, max=3)
            card = hand.gather(1, slot[:, None])[:, 0].clamp(min=0)
            card = torch.where(played, card, torch.zeros_like(card))
            gate, tile, value = model(b, card_ids=card)
            with torch.no_grad():
                rgate, rtile, _ = ref(b, card_ids=card)

            def gate_logq(gl):
                gl = gl.float().clone()
                gl[:, 0] = gl[:, 0] + wait_bias
                gl = gl.masked_fill(~mask, -1e9)
                lp = F.log_softmax(gl, 1)
                lw = lp[:, 0] * (DECIDE / 20)                      # log q_wait = log p_wait ** 0.5
                q_play = (1 - lw.exp()).clamp(min=1e-6)
                ls = F.log_softmax(gl[:, 1:], 1)                   # slot share among playable
                lq = torch.cat([lw[:, None], q_play.log()[:, None] + ls], 1)
                return lq

            lq = gate_logq(gate)
            lq_ref = gate_logq(rgate)
            new = lq.gather(1, a[:, None])[:, 0]
            # tile: log-softmax over the recorded legal candidates
            cands = torch.as_tensor(d["tile_cands"][idx]).long().to(dev)
            cmask = cands >= 0
            tl = tile.float().gather(1, cands.clamp(min=0)) / TILE_TEMP
            tl = tl.masked_fill(~cmask, -1e9)
            ncand = cmask.float().sum(1, keepdim=True).clamp(min=1)
            ptile = (1 - EXPLORE) * F.softmax(tl, 1) + EXPLORE / ncand
            ltile = ptile.clamp(min=1e-12).log().masked_fill(~cmask, -1e9)
            ta = torch.as_tensor(d["tile_a"][idx]).long().to(dev).clamp(min=0)
            new = new + torch.where(played, ltile.gather(1, ta[:, None])[:, 0], torch.zeros_like(new))
            ratio = (new - old).exp()
            pg = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            # keep near the imitation policy (gate distribution, and tile on plays)
            kl_gate = (lq.exp() * (lq - lq_ref)).sum(1).mean()
            rtl = rtile.float().gather(1, cands.clamp(min=0)).masked_fill(~cmask, -1e9)
            # the reference under the same exploration mixture, or the brake
            # would push against the exploration itself
            pref = (1 - EXPLORE) * F.softmax(rtl / TILE_TEMP, 1) + EXPLORE / ncand
            lref_tile = pref.clamp(min=1e-12).log().masked_fill(~cmask, -1e9)
            kl_tile = ((ltile.exp() * (ltile - lref_tile)).masked_fill(~cmask, 0).sum(1)
                       * played.float()).sum() / played.float().sum().clamp(min=1)
            v_loss = F.binary_cross_entropy_with_logits(value[:, 1].float(), ret)
            loss = pg + args.kl * (kl_gate + kl_tile) + 0.5 * v_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats.append((pg.item(), kl_gate.item(), kl_tile.item(), v_loss.item(),
                          (ratio - 1).abs().mean().item()))
    s = np.mean(stats, 0)
    blob["model"] = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(blob, args.out)
    print(json.dumps({"rows": n, "pg": round(s[0], 4), "kl_gate": round(s[1], 4),
                      "kl_tile": round(s[2], 4), "value_bce": round(s[3], 4),
                      "ratio_dev": round(s[4], 4)}), flush=True)


# ----------------------------------------------------------------------------
# loop (royalegym env drives both)
# ----------------------------------------------------------------------------

def loop(args):
    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    rg = Path(sys.executable)
    bb = Path(args.train_python) if args.train_python else rg
    me = str(Path(__file__).resolve())
    # args.json next to each checkpoint so RSAgent reads the right wait_keep
    init_args = Path(args.ref or args.init).with_name("args.json")
    wait_keep = json.loads(init_args.read_text())["wait_keep"] if init_args.exists() else "0.25"
    saved_args = {**vars(args), "wait_keep": wait_keep, "schema_version": 3}
    args_path = run / "args.json"
    existing = sorted(run.glob("it*.pt"))
    if args_path.exists() and existing:
        old = json.loads(args_path.read_text())
        if old.get("schema_version") != 3:
            raise SystemExit("self-play run uses an older training format; "
                             "choose a new --name")
        allowed_to_change = {"iters", "workers", "eval_every", "eval_matches"}
        changed = []
        for key, value in saved_args.items():
            if key in allowed_to_change or key not in old:
                continue
            if str(old[key]) != str(value):
                changed.append(f"{key}: {old[key]!r} -> {value!r}")
        if changed:
            raise SystemExit("self-play run already exists with incompatible settings; "
                             "choose a new --name\n  " + "\n  ".join(changed))
    ckpt = run / "it000.pt"
    if not ckpt.exists():
        import shutil
        shutil.copy(args.init, ckpt)
    args_path.write_text(json.dumps(saved_args, default=str))
    log = run / "log.jsonl"
    start = max([int(p.stem[2:]) for p in run.glob("it*.pt")])
    if start >= args.iters:
        print(json.dumps({"status": "already_complete", "iterations": start,
                          "run": str(run)}), flush=True)
        return 0
    for it in range(start, args.iters):
        ckpt = run / f"it{it:03d}.pt"
        data = run / f"it{it:03d}.npz"
        t0 = time.time()
        reactive_rate = linear_schedule(args.reactive_start, args.reactive_end, it,
                                        args.curriculum_iters)
        opponent_decide = max(1, int(round(linear_schedule(
            args.opponent_decide_start, args.opponent_decide_end, it,
            args.curriculum_iters))))
        c = subprocess.run([str(rg), me, "collect", "--ckpt", str(ckpt), "--opponent", args.opponent,
                            "--matches", str(args.matches), "--workers", str(args.workers),
                            "--seed", str(it + 1), "--out", str(data),
                            "--deck-config", str(args.deck_config),
                            "--reactive-rate", str(reactive_rate),
                            "--general-opponent-rate", str(args.general_opponent_rate),
                            "--opponent-decide", str(opponent_decide),
                            "--terminal-weight", str(args.terminal_weight),
                            "--crown-weight", str(args.crown_weight),
                            "--tower-weight", str(args.tower_weight),
                            ("--randomize-reactive" if args.randomize_reactive
                             else "--no-randomize-reactive")],
                           capture_output=True, text=True)
        if c.returncode != 0:
            print(c.stderr[-2000:], flush=True)
            return 1
        col = json.loads(c.stdout.strip().splitlines()[-1])
        nxt = run / f"it{it + 1:03d}.pt"
        t = subprocess.run([str(bb), me, "train", "--ckpt", str(ckpt), "--ref", args.ref or args.init,
                            "--data", str(data), "--out", str(nxt), "--lr", str(args.lr),
                            "--kl", str(args.kl), "--wait-keep", str(wait_keep),
                            "--seed", str(it + 1)],
                           capture_output=True, text=True)
        if t.returncode != 0:
            print(t.stderr[-2000:], flush=True)
            return 1
        tr = json.loads(t.stdout.strip().splitlines()[-1])
        data.unlink(missing_ok=True)                    # rollouts are cheap to regenerate
        rec = {"iter": it, "reactive_rate": round(reactive_rate, 4),
               "opponent_decide": opponent_decide,
               **{f"collect_{k}": v for k, v in col.items()}, **tr}
        if args.eval_every and (it + 1) % args.eval_every == 0:
            e = subprocess.run([str(rg), str(ROOT / "tools" / "rs_eval.py"), "--ckpt", str(nxt),
                                "--opponent-ckpt", args.eval_opponent, "--matches", str(args.eval_matches),
                                "--workers", str(args.workers), "--delay", "21", "--lead", "-21",
                                "--deck-config", str(args.deck_config), "--json"],
                               capture_output=True, text=True)
            if e.returncode:
                rec["eval_error"] = e.stderr[-500:]
            else:
                try:
                    summary = json.loads(e.stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    rec["eval_error"] = "evaluation returned no JSON summary"
                else:
                    rec["eval"] = summary
                    score = evaluation_score(summary)
                    best_meta = run / "best.json"
                    previous = json.loads(best_meta.read_text()) if best_meta.exists() else None
                    old_score = tuple(previous["score"]) if previous else (-1.0, -999.0)
                    if score > old_score:
                        import shutil
                        shutil.copy2(nxt, run / "best.pt")
                        best = {"checkpoint": nxt.name, "score": list(score),
                                "evaluation": summary}
                        best_meta.write_text(json.dumps(best, indent=2))
                        rec["new_best"] = nxt.name
        rec["seconds"] = round(time.time() - t0)
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--opponent", required=True)
    c.add_argument("--matches", type=int, default=48)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out", required=True)
    c.add_argument("--deck-config", type=Path,
                   default=ROOT / "configs" / "decks" / "hog_26.json")
    c.add_argument("--reactive-rate", type=float, default=0.25)
    c.add_argument("--general-opponent-rate", type=float, default=0.5,
                   help="share of reactive matches using decks from the general replay pool")
    c.add_argument("--opponent-decide", type=int, default=20,
                   help="reactive opponent decision interval in 20 Hz ticks")
    c.add_argument("--randomize-reactive", action=argparse.BooleanOptionalAction, default=True,
                   help="shuffle both starting decks in reactive matches")
    c.add_argument("--terminal-weight", type=float, default=0.7)
    c.add_argument("--crown-weight", type=float, default=0.1)
    c.add_argument("--tower-weight", type=float, default=0.2)
    t = sp.add_parser("train")
    t.add_argument("--ckpt", required=True)
    t.add_argument("--ref", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--kl", type=float, default=0.05)
    t.add_argument("--clip", type=float, default=0.2)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--wait-keep", default="0.25")
    t.add_argument("--seed", type=int, default=0)
    lp = sp.add_parser("loop")
    lp.add_argument("--name", default="sp_v1")
    lp.add_argument("--init", default=str(ROOT / "runs" / "tokens" / "rs_big_hog" / "best.pt"))
    lp.add_argument("--ref", default="",
                    help="model the KL brake pulls toward (default: --init). Set it to the "
                         "imitation model when --init is already a self-play checkpoint")
    lp.add_argument("--opponent", default=str(ROOT / "runs" / "tokens" / "rs_big" / "best.pt"),
                    help="frozen reactive opponent playing the real opponent's deck")
    lp.add_argument("--iters", type=int, default=40)
    lp.add_argument("--matches", type=int, default=48)
    lp.add_argument("--workers", type=int, default=4)
    lp.add_argument("--lr", type=float, default=1e-5)
    lp.add_argument("--kl", type=float, default=0.05)
    lp.add_argument("--reactive-start", type=float, default=0.25,
                    help="reactive-opponent share at iteration zero")
    lp.add_argument("--reactive-end", type=float, default=0.75,
                    help="reactive-opponent share after the curriculum ramp")
    lp.add_argument("--curriculum-iters", type=int, default=20)
    lp.add_argument("--opponent-decide-start", type=int, default=20,
                    help="initial reactive opponent cadence (20 ticks = 1 second)")
    lp.add_argument("--opponent-decide-end", type=int, default=10,
                    help="final reactive opponent cadence (10 ticks = 0.5 seconds)")
    lp.add_argument("--general-opponent-rate", type=float, default=0.5,
                    help="share of reactive matches drawing a deck from the general corpus")
    lp.add_argument("--randomize-reactive", action=argparse.BooleanOptionalAction, default=True)
    lp.add_argument("--terminal-weight", type=float, default=0.7)
    lp.add_argument("--crown-weight", type=float, default=0.1)
    lp.add_argument("--tower-weight", type=float, default=0.2)
    lp.add_argument("--eval-every", type=int, default=5)
    lp.add_argument("--eval-matches", type=int, default=300)
    lp.add_argument("--eval-opponent", default=str(ROOT / "runs" / "tokens" / "rs_bc" / "ep1.pt"),
                    help="the reference every earlier number was measured against")
    lp.add_argument("--deck-config", type=Path,
                    default=ROOT / "configs" / "decks" / "hog_26.json")
    lp.add_argument("--train-python", default="",
                    help="Python executable with CUDA torch; default is the current interpreter")
    args = ap.parse_args()
    return {"collect": collect, "train": train, "loop": loop}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
