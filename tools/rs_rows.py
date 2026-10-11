"""Training rows from IL_Replay re-driven through RoyaleSim, in the same column
format as sim/replay_rows.py, so sim/train_tokens.py trains on either.

    .venvs/royalegym/Scripts/python.exe tools/rs_rows.py --others 20000 --workers 7

Why a second generator: on the same 2,000 matches RoyaleSim's endings agree
with the recording far more often than ours (winner 62.6% vs 49.5%, exact
crowns 49.0% vs 24.2%; reports/redrive_fidelity.md), and it refuses almost no
recorded play, so the boards a row shows are much closer to what the player
really saw.

Frame: every row is from the acting player's view, as the bottom player with
y growing toward its own king (the same as sim/replay_rows.py). RoyaleSim's
blue owns low y, so blue is reflected in y and red in x; both are reflections,
so left and right mean the same thing to either seat.

Ids: units are RoyaleSim `unit_type` + 2, cards are catalogue ids + 2 (0 pad,
1 unknown). `data/rs_rows/vocab.json` records both lists.

Outcome and trust as in sim/replay_rows.py: tower swing over the next 10 s;
rows after the first refused play, or after the engine has taken more crowns
than the recording allows, are marked untrusted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mode

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from sim.deck_config import (DEFAULT_DECK_CONFIG, ability_button_for_event,
                             ability_buttons, load_deck_config)  # noqa: E402
from royalesim_redrive import CORPUS, base_key, deal_order  # noqa: E402

OUT = ROOT / "data" / "rs_rows"
MAX_UNITS, UF = 48, 7
HORIZON, WAIT_EVERY, CHUNK = 200, 20, 10
# Kept for older callers. New code selects the target through --deck-config.
HOG26 = frozenset(load_deck_config(DEFAULT_DECK_CONFIG).engine_cards)
MODES = {"Ranked", "Ladder", "1v1 Battle", "Grand Challenge", "Classic Challenge"}
F = None   # field indexes, filled on first use
# --live-compat: unit tokens carry only what the live memory reader provides.
# The reader reports the card that put a unit down (a Witch's skeleton reads as
# the Witch), hit points and max hit points, side and position; not RoyaleSim's
# unit_type, not a deploy timer, not a flying flag. Rows built with this flag
# use: id = creating card, flying / building from that card's catalogue row,
# and log(max hp) in place of the deploy flag, so a model trained on them sees
# in training exactly what it will see live.
LIVE = False
CARD_FLY: dict = {}
CARD_BUILDING: dict = {}


def split_of(tag: str) -> str:
    h = int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 5 else "val" if h < 10 else "train"


def _setup():
    global F
    import royalesim
    import royaleimitate
    names = json.loads((Path(royaleimitate.__file__).with_name("replay_cards.json"))
                       .read_text(encoding="utf-8"))
    b = royalesim.Battle(card_names=None, slot_of_k=[[0, 1, 2], [0, 1, 2]])
    cat = json.loads(b.catalogue_json())
    ids = {row[0]: i for i, row in enumerate(cat)}
    F = {k: i for i, k in enumerate(royalesim.ENTITY_FIELDS)}
    for i, row in enumerate(cat):
        CARD_FLY[i] = bool(row[5])
        CARD_BUILDING[i] = row[8] == "BUILDING"
    return royalesim, names, ids, cat


def own(x_sub, y_sub, team, sub):
    """RoyaleSim sub-tiles -> the acting player's frame, millitiles, bottom = high y."""
    x, y = x_sub / sub, y_sub / sub
    return (x, 32_000 - y) if team == 0 else (18_000 - x, y)


def towers_vec(state, team):
    out = []
    for t in (team, 1 - team):
        hp, mx = state["players"][t]["tower_hp"], state["players"][t]["tower_max_hp"]
        # slots: 0 king, 1 world-left, 2 world-right. Red is reflected in x, so
        # its own left is world right.
        lr = (1, 2) if team == 0 else (2, 1)
        out += [hp[lr[0]] / max(1, mx[lr[0]]), hp[lr[1]] / max(1, mx[lr[1]]),
                hp[0] / max(1, mx[0])]
    return np.asarray(out, dtype=np.float32)


def obs_row(state, team, sub, mem, tick, live=None):
    live = LIVE if live is None else live
    ents = [e for e in state["entities"] if e[F["tower_slot"]] < 0 and e[F["hp"]] > 0]
    king = next((e for e in state["entities"]
                 if e[F["team"]] == team and e[F["tower_slot"]] == 0), None)
    # A destroyed king (a lookahead copy can end the battle) has no entity;
    # sort from where it stands instead.
    kx, ky = ((king[F["x"]], king[F["y"]]) if king is not None
              else (9000 * 18, (3000 if team == 0 else 29000) * 18))
    ents.sort(key=lambda e: (e[F["x"]] - kx) ** 2 + (e[F["y"]] - ky) ** 2)
    ids = np.zeros(MAX_UNITS, dtype=np.int16)
    feats = np.zeros((MAX_UNITS, UF), dtype=np.float16)
    for k, e in enumerate(ents[:MAX_UNITS]):
        ox, oy = own(e[F["x"]], e[F["y"]], team, sub)
        if live:
            card = e[F["card_id"]]
            ids[k] = card + 2
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]),
                        1.0 if CARD_FLY.get(card) else 0.0,
                        1.0 if CARD_BUILDING.get(card) else 0.0,
                        float(np.log1p(max(0, e[F["max_hp"]]))) / 10.0)
        else:
            ids[k] = e[F["unit_type"]] + 2
            feats[k] = (1.0 if e[F["team"]] == team else 0.0, ox / 18_000, oy / 32_000,
                        e[F["hp"]] / max(1, e[F["max_hp"]]), 1.0 if e[F["flying"]] else 0.0,
                        1.0 if e[F["kind"]] == 1 else 0.0,
                        1.0 if e[F["deploy_ticks"]] > 0 else 0.0)
    p = state["players"][team]
    hand = [c + 2 for c in p["hand"][:4]] + [0] * (4 - len(p["hand"][:4]))
    opp = 1 - team
    seen = sorted(mem["seen"][opp])[:8]
    last = mem["last"][opp][-4:]
    ability_ids = np.zeros(3, dtype=np.int16)
    ability_feats = np.zeros((3, 4), dtype=np.float16)
    for k, ability in enumerate(p.get("abilities", [])[:3]):
        available, spent, cost, card_id, cooldown = ability[:5]
        ability_ids[k] = int(card_id) + 2 if int(card_id) >= 0 else 0
        ability_feats[k] = (float(available), float(spent), float(cost) / 10.0,
                            min(1.0, float(cooldown) / 600.0))
    t_ms = tick * 50
    return {
        "tick": tick, "side": 1 if team == 0 else -1, "hand_exact": True,
        "unit_ids": ids, "unit_feats": feats, "towers": towers_vec(state, team),
        "scalars": np.asarray([p["elixir_milli"] / 10_000, min(1.0, t_ms / 300_000),
                               1.0 if t_ms >= 120_000 else 0.0,
                               1.0 if t_ms >= 180_000 else 0.0], dtype=np.float32),
        "hand": np.asarray(hand, dtype=np.int16),
        "next": (p["next_card"] + 2) if p.get("next_card", -1) >= 0 else 0,
        "opp_seen": np.asarray(seen + [0] * (8 - len(seen)), dtype=np.int16),
        "opp_last": np.asarray([c for c, _ in last] + [0] * (4 - len(last)), dtype=np.int16),
        "opp_last_age": np.asarray([min(1.0, (tick - t) / 400) for _, t in last]
                                   + [1.0] * (4 - len(last)), dtype=np.float16),
        "ability_ids": ability_ids, "ability_feats": ability_feats,
    }


def rows_for(payload: dict, ctx, target_config=None):
    royalesim, names, ids, cat = ctx
    target_config = target_config or load_deck_config(DEFAULT_DECK_CONFIG)
    sub = royalesim.SUBTILE_PER_MILLITILE
    battle = payload["battle"]
    decks, forms, levels = [], [], []
    for side in ("team", "opponent"):
        players = battle[side]["players"]
        if len(players) != 1:
            return None, "not 1v1"
        deck, form, level = [], [], []
        for card in players[0]["deck"]:
            key, f = base_key(card["card_key"])
            name = names.get(key)
            if name is None or name not in ids:
                return None, "card"
            deck.append(name)
            form.append(f)
            level.append(int(card.get("level") or 11))
        decks.append(deck)
        forms.append(form)
        levels.append(level)
    buttons = [ability_buttons(decks[s], forms[s], cat) for s in (0, 1)]
    events = sorted((e for e in payload["events"]
                     if e["kind"] in ("play_card", "activate_ability")),
                    key=lambda e: (e["replay_tick_20hz"], e["source_index"]))
    plays = {0: [], 1: []}
    for e in events:
        if e["kind"] != "play_card":
            continue
        seat = 0 if e["side"] == "team" else 1
        name = names.get(base_key(e["card_key"])[0])
        u = e["coordinates"]["native_world_units"]
        plays[seat].append((int(e["replay_tick_20hz"]), name, int(u["x"]), int(u["y"])))
    ys = sorted(y for _, _, _, y in plays[0]) or [0]
    team_seat = 0 if ys[len(ys) // 2] < 16_000 else 1
    dealt = []
    for seat in (0, 1):
        order = deal_order(decks[seat], [c for _, c, _, _ in plays[seat]])
        if order is None:
            return None, "deal"
        dealt.append(order)
    perm = [[decks[s].index(c) for c in dealt[s]] for s in (0, 1)]
    rs = {0: team_seat, 1: 1 - team_seat}
    seat_of = {rs[0]: 0, rs[1]: 1}
    tower = max(1, min(16, mode([lv for s in levels for lv in s])))
    b = royalesim.Battle(card_names=None, slot_of_k=[[0, 1, 2], [0, 1, 2]])
    b.reset(seed=0, decks=[[ids[c] for c in dealt[seat_of[t]]] for t in (0, 1)],
            shuffle=0, start_tick=0, elixir_milli=[6000, 6000], tower_hp=None, spawns=[],
            forms=[[forms[seat_of[t]][i] for i in perm[seat_of[t]]] for t in (0, 1)],
            levels=[[levels[seat_of[t]][i] for i in perm[seat_of[t]]] for t in (0, 1)],
            tower_levels=[tower, tower])
    rec_crowns = {rs[0]: battle["team"]["crowns"], rs[1]: battle["opponent"]["crowns"]}
    result = str(battle.get("result", "")).lower()
    won_team = 1.0 if result in ("victory", "win") else 0.0 if result in ("defeat", "loss") else 0.5
    won = {rs[0]: won_team, rs[1]: 1.0 - won_team}
    queue = []
    for e in events:
        seat = 0 if e["side"] == "team" else 1
        tick_e = int(e["replay_tick_20hz"])
        source = int(e.get("source_index", 0))
        if e["kind"] == "activate_ability":
            button = ability_button_for_event(e, buttons[seat], names)
            if button is not None:
                queue.append((tick_e, source, rs[seat], "ability", button, 0, 0))
        else:
            name = names.get(base_key(e["card_key"])[0])
            u = e["coordinates"]["native_world_units"]
            queue.append((tick_e, source, rs[seat], "play", name, int(u["x"]), int(u["y"])))
    queue.sort(key=lambda q: (q[0], q[1]))
    play_ticks = {0: [q[0] for q in queue if q[2] == 0],
                  1: [q[0] for q in queue if q[2] == 1]}
    mem = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}
    rows, track = [], []
    cut = 10 ** 9
    tick, i = 0, 0
    end = min((queue[-1][0] if queue else 0) + HORIZON, 6000)
    while tick <= end:
        state = json.loads(b.state_json())
        if state["game_over"]:
            track.append((tick, towers_vec(state, 0)))
            break
        track.append((tick, towers_vec(state, 0)))
        for t in (0, 1):
            if state["players"][t]["crowns"] > rec_crowns[t] and cut > tick:
                cut = tick
        due = []
        while i < len(queue) and queue[i][0] <= tick:
            due.append(queue[i])
            i += 1
        cmds = []
        acting = set()
        for (_, _, team, kind, action, x, y) in due:
            acting.add(team)
            row = obs_row(state, team, sub, mem, tick)
            if kind == "ability":
                row["action"] = (0, 0, 0)
                row["act_gate"] = 5 + int(action)
                cmds.append((team, 4 + int(action), 0, 0))
            else:
                card = action
                cid = ids[card]
                hand = state["players"][team]["hand"]
                if team_seat == 1:
                    x, y = 18_000 - x, 32_000 - y
                ox, oy = own(x * sub, y * sub, team, sub)
                row["action"] = (cid + 2, int(min(17, max(0, ox // 1000))),
                                 int(min(31, max(0, oy // 1000))))
                row["act_gate"] = 1 + hand.index(cid) if cid in hand else -1
                if cid in hand:
                    cmds.append((team, hand.index(cid), x * sub, y * sub))
                elif cut > tick:
                    cut = tick
                mem["seen"][team].add(cid + 2)
                mem["last"][team].append((cid + 2, tick))
            rows.append(row)
        if tick % WAIT_EVERY == 0:
            for team in (0, 1):
                soon = any(tick < t <= tick + 10 for t in play_ticks[team])
                if team not in acting and not soon and tick > 0:
                    row = obs_row(state, team, sub, mem, tick)
                    row["action"] = (0, 0, 0)
                    row["act_gate"] = 0
                    rows.append(row)
        nxt = queue[i][0] if i < len(queue) else end + 1
        step = max(1, min(nxt, (tick // WAIT_EVERY + 1) * WAIT_EVERY) - tick)
        res = b.step(cmds, step)
        for r in res:
            if royalesim.DEPLOY_REASONS[r[1]] != "OK" and cut > tick:
                cut = tick
        tick += step
    ticks = np.asarray([t for t, _ in track])
    tv = np.stack([v for _, v in track])            # blue-frame [own L,R,K, enemy L,R,K]
    blue_total, red_total = tv[:, :3].sum(1), tv[:, 3:].sum(1)
    for row in rows:
        team = 0 if row["side"] == 1 else 1
        a = int(np.searchsorted(ticks, row["tick"]))
        z = int(np.searchsorted(ticks, row["tick"] + HORIZON))
        a, z = min(a, len(ticks) - 1), min(z, len(ticks) - 1)
        own_lost = (blue_total if team == 0 else red_total)
        opp_lost = (red_total if team == 0 else blue_total)
        row["outcome"] = float((opp_lost[a] - opp_lost[z]) - (own_lost[a] - own_lost[z]))
        row["trusted"] = row["tick"] + HORIZON <= cut
        row["won"] = won[team]
        replay_side = "team" if seat_of[team] == 0 else "opponent"
        row["target"] = target_config.matches_replay_cards(
            battle[replay_side]["players"][0]["deck"])
    return rows, "ok"


FIELDS = ("tick", "side", "hand_exact", "unit_ids", "unit_feats", "towers", "scalars",
          "hand", "next", "opp_seen", "opp_last", "opp_last_age", "ability_ids",
          "ability_feats", "outcome", "trusted", "won", "target", "act_gate")


def build(batch_id, payload_texts, live=False, out_dir=None, deck_config=None):
    global LIVE, OUT
    LIVE = live
    if out_dir:
        OUT = Path(out_dir)
    ctx = _setup()
    target_config = load_deck_config(deck_config)
    rows, tags, skipped = [], [], {}
    for text in payload_texts:
        p = json.loads(text)
        tag = p["source"]["replay_tag"]
        try:
            r, why = rows_for(p, ctx, target_config)
        except Exception as exc:
            r, why = None, f"error {type(exc).__name__}"
        if r is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        rows += r
        tags += [tag] * len(r)
    if rows:
        data = {k: np.stack([np.asarray(r[k]) for r in rows]) for k in FIELDS}
        act = np.asarray([r["action"] for r in rows], dtype=np.int16)
        data["act_card"], data["act_x"], data["act_y"] = act[:, 0], act[:, 1], act[:, 2]
        data["match"] = np.asarray(tags)
        data["split"] = np.asarray([split_of(t) for t in tags])
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"shard-{batch_id:04d}.npz", **data)
    return batch_id, len(rows), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--others", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--limit-parts", type=int, default=0)
    ap.add_argument("--live-compat", action="store_true",
                    help="unit tokens limited to what the live reader provides")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--deck-config", type=Path, default=DEFAULT_DECK_CONFIG,
                    help="target deck JSON; exact forms are matched when match_forms is true")
    ap.add_argument("--seed", type=int, default=0,
                    help="deterministic reservoir sample of non-target matches")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing shard-*.npz and vocab.json in --out")
    args = ap.parse_args()
    global OUT, LIVE
    LIVE = args.live_compat
    if args.out:
        OUT = args.out
    import pyarrow.parquet as pq
    royalesim, names, ids, cat = _setup()
    target_config = load_deck_config(args.deck_config)
    existing = sorted(OUT.glob("shard-*.npz"))
    if existing and not args.overwrite:
        raise SystemExit(f"{OUT} already contains {len(existing)} shards; choose a new --out "
                         "or pass --overwrite to rebuild it without mixing datasets")
    if args.overwrite:
        for path in existing:
            path.unlink()
        (OUT / "vocab.json").unlink(missing_ok=True)
    t0 = time.time()
    target, others = [], []
    rng = np.random.default_rng(args.seed)
    n_other = 0
    parts = sorted((CORPUS / "replays").glob("part-*.parquet"))
    if args.limit_parts:
        parts = parts[:args.limit_parts]
    for part in parts:
        for r in pq.read_table(part, columns=["replay_tag", "game_mode", "payload_json"]).to_pylist():
            if r["game_mode"] not in MODES:
                continue
            p = json.loads(r["payload_json"])
            ok = True
            for side in ("team", "opponent"):
                pl = p["battle"][side]["players"]
                if len(pl) != 1:
                    ok = False
                    break
                if any(names.get(base_key(c["card_key"])[0]) is None for c in pl[0]["deck"]):
                    ok = False
                    break
            if not ok:
                continue
            p.setdefault("source", {})["replay_tag"] = r["replay_tag"]
            text = json.dumps(p)
            if target_config.target_seat(p["battle"]) is not None:
                target.append(text)
            elif args.others:
                # Reservoir sampling keeps a representative fixed-size sample
                # while the full corpus is scanned for every rare target deck.
                n_other += 1
                if len(others) < args.others:
                    others.append(text)
                else:
                    replace = int(rng.integers(n_other))
                    if replace < args.others:
                        others[replace] = text
    picked = target + others
    print(f"selected {len(target):,} {target_config.label} + {len(others):,} others "
          f"({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    unit_types = json.loads(royalesim.Battle(card_names=None, slot_of_k=[[0, 1, 2], [0, 1, 2]])
                            .unit_types_json())
    (OUT / "vocab.json").write_text(json.dumps({
        "n_cards": len(cat) + 2,
        "n_units": (len(cat) + 2) if LIVE else (len(unit_types) + 2),
        "live_compat": LIVE,
        "cards": {row[0]: i + 2 for i, row in enumerate(cat)},
        "units": {u: i + 2 for i, u in enumerate(unit_types)},
        "ability_actions": 3, "ability_features": 4,
        "target_deck": target_config.metadata(),
        "engine": "royalesim", "version": getattr(royalesim, "__version__", "?")}, indent=2))
    batches = [picked[i:i + args.batch] for i in range(0, len(picked), args.batch)]
    total, skipped = 0, {}
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(build, k, b, LIVE, str(OUT), str(target_config.source))
                for k, b in enumerate(batches)]
        for n, f in enumerate(as_completed(futs), 1):
            bid, rows, sk = f.result()
            total += rows
            for k, v in sk.items():
                skipped[k] = skipped.get(k, 0) + v
            if n % 10 == 0 or n == len(batches):
                print(f"  {n}/{len(batches)} shards, {total:,} rows, skipped {skipped} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"done: {total:,} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
