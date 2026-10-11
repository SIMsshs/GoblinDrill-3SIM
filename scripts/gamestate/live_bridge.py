"""Live bridge, dry run: reader frame -> live-compatible observation -> model
decision, printed. It never taps.

    py -3.12 scripts/gamestate/live_bridge.py --ckpt runs/tokens/rs_live/best.pt

Run it during a friendly match the owner starts; it prints what the bot WOULD
play each half second next to what the reader saw. Tapping is a separate,
later step and needs the owner's go-ahead.

The observation is built exactly as `tools/rs_rows.py --live-compat` builds
training rows, so the model sees live what it saw in training:
    unit id = creating card (native id -> RoyaleSim id via card_map.json),
    own / x / y / hp fraction / flying / building (card catalogue) / log max hp,
    towers as hit point fractions, own elixir, clock, own hand and next card,
    the opponent's seen cards and last four plays (from units appearing).

Open items, each needing a live battle to check (frame.validate covers 1-3):
 1. which reader side is ours, and which end of the board it owns
    (taken from our king tower's y; checked against the first plays);
 2. tower entities: told apart from troops by position and max hp until the
    tower card ids are known;
 3. hand: the reader gives deck indexes; the deck's card ids come from the
    avatar chain (root +0x30 ... entry +0x10 -> data +0x40) that the native
    reader does not walk yet - until then the deck is passed with --deck;
 4. evolution ids (13xxxxxx) are not in card_map.json. The native reader now
    emits each unit's character name (11th field; entity +0x48 -> data name,
    e.g. "Skeleton_EV1"). Those are the client's names, and RoyaleSim's
    `unit_types_json()` uses the same ones, so the unit-type model (`rs_hog`,
    the stronger one) can run live: unit id = unit_types.index(name) + 2.
    Check in the first battle that every name read is in that list;
 5. delay: the bot decides on a frame ~100 ms old and a tap lands ~1.1 s
    later (RoyaleSim measured 21-22 ticks); the board should be advanced by
    that much before deciding - not done here yet.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gamestate.memread import GameMemory, _sh  # noqa: E402
from sim.deck_config import DEFAULT_DECK_CONFIG, load_deck_config  # noqa: E402

CARD_MAP = json.loads((Path(__file__).with_name("card_map.json")).read_text())["cards"]
TOWER_MIN_HP = 2000          # crown towers; no troop starts this high except a few tanks
MAX_UNITS, UF = 48, 7


class LiveObs:
    def __init__(self, deck_rs_ids, catalogue):
        self.deck = deck_rs_ids                       # our 8 cards, RoyaleSim ids
        self.cat = catalogue
        self.seen_opp, self.last_opp = set(), []
        self.known_units = set()

    def from_frame(self, f, my_side):
        units = f["u"]
        my_king = [u for u in units if u[3] == my_side and u[9] >= TOWER_MIN_HP]
        low = (min(u[5] for u in my_king) < 16_000) if my_king else True
        def own(x, y):                                # our end -> bottom, high y
            return (x, 32_000 - y) if low else (18_000 - x, y)
        troops = []
        for addr, cat, kind, side, x, y, card, level, hp, mx in units:
            entry = CARD_MAP.get(str(card))
            if entry is None:
                continue                              # towers, evo ids, unknowns
            troops.append((addr, side, x, y, entry["rs_id"], hp, mx))
            if side != my_side and addr not in self.known_units:
                self.seen_opp.add(entry["rs_id"] + 2)
                self.last_opp.append((entry["rs_id"] + 2, f["t"]))
            self.known_units.add(addr)
        kx, ky = own(9000, 3000 if low else 29_000)
        troops.sort(key=lambda t: (t[2] - kx) ** 2 + (t[3] - ky) ** 2)
        ids = np.zeros(MAX_UNITS, np.int64)
        feats = np.zeros((MAX_UNITS, UF), np.float32)
        for k, (addr, side, x, y, rs, hp, mx) in enumerate(troops[:MAX_UNITS]):
            ox, oy = own(x, y)
            row = self.cat[rs]
            ids[k] = rs + 2
            feats[k] = (1.0 if side == my_side else 0.0, ox / 18_000, oy / 32_000,
                        hp / max(1, mx), 1.0 if row[5] else 0.0,
                        1.0 if row[8] == "BUILDING" else 0.0, math.log1p(max(0, mx)) / 10)
        return ids, feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--deck", default="",
                    help="deck in the deck-screen order (left to right, top row first); "
                         "overrides --deck-config")
    ap.add_argument("--deck-config", type=Path, default=DEFAULT_DECK_CONFIG)
    ap.add_argument("--side", type=int, default=1,
                    help="our player record's side; 1 in the 2026-10-10 friendlies, UNVERIFIED always")
    ap.add_argument("--seconds", type=int, default=240)
    args = ap.parse_args()
    print("DRY RUN: decisions are printed, nothing is tapped.")
    # The decision code mirrors tools/rs_eval.RSAgent.act, so live and the
    # 300-match screens use the same gate, wait bias and tile head.
    sys.path.insert(0, str(ROOT / "tools"))
    import rs_rows as RR
    from rs_eval import DECIDE, RSAgent, _unit_names
    royalesim, names, ids, cat = RR._setup()
    sub = royalesim.SUBTILE_PER_MILLITILE
    unit_types = _unit_names()
    deck_config = load_deck_config(args.deck_config)
    deck_names = args.deck.split(",") if args.deck else list(deck_config.engine_cards)
    deck_forms = ([0] * 8 if args.deck else list(deck_config.forms))
    deck_rs = [ids[c] for c in deck_names]
    agent = RSAgent(args.ckpt)
    torch = agent.torch
    mem = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}
    known = set()
    dev = GameMemory()
    reader = "/data/local/tmp/hcr_reader"
    proc = subprocess.Popen(
        [r"C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe",
         "-s", dev.serial, "exec-out",
         f"HCR_SIDE={args.side} {reader} {dev.pid} {dev.libg:x} 100"],
        stdout=subprocess.PIPE, text=True)
    t0, last_decide = time.time(), -10**9
    for line in proc.stdout:
        if time.time() - t0 > args.seconds:
            break
        try:
            f = json.loads(line)
        except json.JSONDecodeError:
            print(f"reader said (not JSON): {line.rstrip()!r}")
            continue
        if f.get("t", -1) < 0 or "me" not in f:
            continue
        state = rs_state_from_frame(f, args.side, deck_rs, unit_types, cat, sub, deck_forms)
        team = next((t for t in (0, 1) if state["players"][t]["hand"]), None)
        if team is None:
            continue
        # Opponent memory from public information only: units appearing on their side.
        for e in state["entities"]:
            if e[1] != team and e[0] not in known and e[3] >= 0:
                mem["seen"][1 - team].add(e[3] + 2)
                mem["last"][1 - team].append((e[3] + 2, f["t"]))
            known.add(e[0])
        if f["t"] - last_decide < DECIDE:
            continue
        last_decide = f["t"]
        row = RR.obs_row(state, team, sub, mem, f["t"], live=agent.live)
        b = agent.batch(row)
        with torch.no_grad():
            gate, _, _ = agent.model(b, card_ids=b["hand"][:, 0].clamp(min=0))
        g = gate[0].float().numpy()
        g[0] += agent.wait_bias
        # The reader does not yet expose the live ability button state or its
        # calibrated screen coordinate. Never guess a paid external tap.
        if len(g) > 5:
            g[5:] = -1e9
        p = state["players"][team]
        for slot in range(4):
            if slot >= len(p["hand"]) or p["elixir_milli"] < p["hand_costs"][slot] * 1000:
                g[1 + slot] = -1e9
        pr = np.exp(g - g.max())
        pr /= pr.sum()
        hand_names = [cat[c][0] for c in p["hand"]]
        head = f"t={f['t']:5d} elixir {p['elixir_milli'] / 1000:4.1f} hand {hand_names}"
        if g[1:].max() < -1e8 or agent.rng.random() < float(pr[0]) ** (DECIDE / 20):
            print(f"{head}  -> wait (p_wait {pr[0]:.2f})")
            continue
        slot = int(np.argmax(g[1:]))
        with torch.no_grad():
            _, tile, _ = agent.model(b, card_ids=b["hand"][:, slot])
        t = int(np.argmax(tile[0].float().numpy()))
        print(f"{head}  -> PLAY {hand_names[slot]} at tile ({t % 18},{t // 18}), "
              f"own frame, row 31 = our back line (p {pr[1 + slot]:.2f}, placement unchecked)")
    proc.terminate()
    return 0




# ----------------------------------------------------------------------------
# Reader frame -> RoyaleSim-shaped state, so the tested code paths run live:
# `tools/rs_rows.obs_row` builds the model input from it, and
# `tools/rs_eval.RSAgent.rebuilt` rebuilds a battle from it for the look-ahead
# (`--lead -21`: the best public-only configuration, 47.0% vs 32.0% without).
#
# UNVALIDATED until a battle is read. Assumptions to check in the first
# friendly, each one printed by `--check-frame`:
#   * a tower is a body whose name contains "Tower" (or, failing names, a
#     body with max hp >= TOWER_MIN_HP at a tower position); king = the one
#     on the centre line; left / right by x;
#   * reader x / y are native world units (1000 per tile), RoyaleSim's
#     sub-tile = native * SUBTILE_PER_MILLITILE;
#   * our hand comes as deck indexes; the deck's RoyaleSim ids are passed in
#     (`--deck`) until the reader walks the avatar chain.
# ----------------------------------------------------------------------------

_FORM_SUFFIXES = ("_EV1", "_EV2", "Hero")
_CARD_BY_NAME = {}


def card_of_form(name, catalogue):
    """RoyaleSim card id for an evolved or hero unit, from its character name.

    Evolutions and heroes have their own native card ids (13000096 for an
    evolved Cannon, 203000014 for the Musketeer hero, seen in the 2026-10-10
    friendly) that card_map.json does not list, so these units were dropped
    from the rebuild and the bot never saw a hero Musketeer or Ice Golem. The
    name is the base unit's name plus a suffix ("Cannon_EV1",
    "MusketeerHero"); the card is the one whose own unit that base is.
    """
    if not _CARD_BY_NAME:
        for i, row in enumerate(catalogue):
            _CARD_BY_NAME[row[0].lower()] = i
        sys.path.insert(0, str(ROOT / "tools"))
        from rs_eval import _spawn_tables, _unit_names
        prim, by_unit = _spawn_tables()
        for unit, card in by_unit.items():
            _CARD_BY_NAME.setdefault(_unit_names()[unit].lower(), card)
    base = name
    for suf in _FORM_SUFFIXES:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    if base == name:
        return -1
    return _CARD_BY_NAME.get(base.lower(), -1)


def rs_state_from_frame(frame, my_side, deck_rs, unit_types, catalogue, sub, deck_forms=None):
    """A dict with the keys rs_rows.obs_row and RSAgent.rebuilt read."""
    ut_index = {name: i for i, name in enumerate(unit_types)}
    teams = {}                                  # reader side -> RoyaleSim team
    units = frame["u"]
    def is_tower(u):
        name = u[10] if len(u) > 10 else ""
        return ("Tower" in name) or (u[9] >= TOWER_MIN_HP and abs(u[4] - 9000) in (0, 5500))
    towers = [u for u in units if is_tower(u)]
    # RoyaleSim team 0 owns low y. Our side's king tells which end is ours.
    mine_low = any(u[3] == my_side and u[5] < 16_000 for u in towers)
    teams[my_side] = 0 if mine_low else 1
    teams[1 - my_side] = 1 - teams[my_side]
    entities, tower_hp = [], {0: [0, 0, 0], 1: [0, 0, 0]}
    tower_max = {0: [1, 1, 1], 1: [1, 1, 1]}
    for u in units:
        addr, cat, kind, side, x, y, card, level, hp, mx = u[:10]
        name = u[10] if len(u) > 10 else ""
        team = teams.get(side)
        if team is None:
            continue
        if is_tower(u):
            slot = 0 if abs(x - 9000) < 2000 else (1 if x < 9000 else 2)
            tower_hp[team][slot], tower_max[team][slot] = max(0, hp), max(1, mx)
            continue
        unit_type = ut_index.get(name, -1)
        # Only characters are units. Spells and their effects (the Log's
        # roller, Ice Spirits' freeze) are entities with no character name;
        # RoyaleSim keeps those out of its unit list, so the model never saw
        # them as units, and counting them logged one Log as two plays.
        if unit_type < 0:
            continue
        entry = CARD_MAP.get(str(card))
        rs_card = entry["rs_id"] if entry else card_of_form(name, catalogue)
        if rs_card < 0:
            continue
        row = [0] * 28
        # kind 1 = building, as RoyaleSim sets it; the unit-type model reads it
        # as the "building" feature (it was always 0 live before 2026-10-10).
        kind = 1 if rs_card >= 0 and catalogue[rs_card][8] == "BUILDING" else 0
        row[0], row[1], row[2], row[3], row[4] = addr, team, kind, rs_card, -1
        row[5], row[6], row[7], row[8] = x * sub, y * sub, hp, mx
        row[10] = bool(catalogue[rs_card][5]) if rs_card >= 0 else False
        row[21], row[27] = level, unit_type
        entities.append(row)
    me = frame.get("me") or [my_side, 0, [], []]
    hand_idx = me[2] if len(me) > 2 else []
    hand = [deck_rs[i] for i in hand_idx if 0 <= i < len(deck_rs)]
    cyc = me[3] if len(me) > 3 else []
    players = [None, None]
    for team in (0, 1):
        players[team] = {"team": team, "tower_hp": tower_hp[team], "tower_max_hp": tower_max[team],
                         "elixir_milli": 0, "hand": [], "hand_costs": [], "deck": [],
                         "forms": [], "abilities": [], "next_card": -1, "crowns": 0}
    t_me = teams[my_side]
    players[t_me].update(
        elixir_milli=int(me[1] / 10) if len(me) > 1 else 0,     # reader raw: 1e-4 elixir
        hand=hand, hand_costs=[catalogue[c][2] for c in hand], deck=list(deck_rs),
        forms=list(deck_forms or [0] * len(deck_rs)),
        next_card=deck_rs[cyc[0]] if cyc and 0 <= cyc[0] < len(deck_rs) else -1)
    return {"tick": frame["t"], "entities": entities, "players": players,
            "spells": [], "projectiles": [], "game_over": False}


if __name__ == "__main__":
    sys.exit(main())
