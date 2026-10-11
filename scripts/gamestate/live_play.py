"""Live play: reader frame -> model -> taps on instance 3, battle after battle.

    .venvs\\royalegym\\Scripts\\python.exe scripts\\gamestate\\live_play.py ^
        --ckpt runs\\tokens\\drill_target_bc\\best.pt

Between battles it navigates with the old bot's screen logic
(scripts/gamestate/nav.py, ported from scripts/cr_bot.py): Battle in the
lobby, the end screen's button, chest / card-reveal screens, connection-lost
RELOAD, popup close buttons, and an app relaunch after 75 s stuck. It plays every battle
until Ctrl+C or a file named STOP in the repo root (--no-queue: you press
Battle yourself).
After each battle it prints the result and the running record, kept in
reports/live_play/record.json. Every tap, every opponent play and every result
is logged to reports/live_play/<time>.jsonl.

Ladder play was authorised by the owner on 2026-10-10.

Decision = the evaluator's (tools/rs_eval.RSAgent.act) with the measured live
configuration: the board is rebuilt from what is on screen and run 21 ticks
ahead (`--lead -21`, 47.0% vs 32.0% without, 300 matches) before deciding,
because a tap lands about 1.1 s after it is sent. The placement is checked in
the rebuilt battle (`check_deploy`), so an illegal tile is never tapped.

Added here, not measured in the evaluator: at 9.5+ elixir the agent does not
wait if a card is affordable (the dry run sat at 10 elixir for half a minute).

Instance safety: scripts/emulator.verify() refuses anything that is not the
Clash Royale instance; the Clash of Clans instance is never touched.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import threading
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tools")]

import emulator  # noqa: E402
from gamestate.live_bridge import CARD_MAP, card_of_form, rs_state_from_frame  # noqa: E402
from gamestate.memread import GameMemory  # noqa: E402
from gamestate.nav import Navigator  # noqa: E402
from sim.deck_config import DEFAULT_DECK_CONFIG, load_deck_config  # noqa: E402

ADB = r"C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe"
CARD_CENTRES = ((334, 1711), (538, 1711), (742, 1711), (946, 1711))   # mumu_overnight_bot
# Grid tile -> MuMu 1080x1920 pixel, copied from scripts/brain/arena.py (measured
# against the live emulator there; importing brain pulls in PIL and the old bot).
_X0, _XS, _Y0, _YS = 103.5, 51.0, 171.9, 41.4
PLAY_WINDOW = 20          # ticks: same card, same opponent play
DECIDE_LIVE = 1           # ticks: decide on every new frame (the evaluator used 10)
PENDING_TICKS = 60        # a sent play not seen after 3 s is assumed lost
# Ticks from the tap decision to the troop on the board, MEASURED live on
# 2026-10-10: 25-27 over a session (RoyaleSim's client figure was 21, plus the
# tap itself). The look-ahead runs this far.
LEAD = 26
FROZEN_FRAMES = 60            # tick unchanged this many frames (3 s) = battle over
CAP_ELIXIR = 9500         # milli; never sit here with an affordable card
FILLER = ["Knight", "Archer", "Giant", "Fireball", "Arrows", "Minions", "Valkyrie", "Goblins"]


# Wrong-lane guard (hand-written rule, not learned). Live logs of 2026-10-09:
# 24% of defensive plays within 3 s of an opponent's play went to the other
# lane, e.g. a Cannon across from a Royal Giant. When one lane clearly holds
# the threat, defensive cards may only go to that lane or the centre.
GUARD_CARDS = {"Cannon"}   # Ice Golem left out: pros pull with it into the other lane 17% of the time
GUARD_MIN_HP = 1200       # enemy hp on our side of the river before the rule acts
GUARD_SHARE = 0.75        # share of that hp in one lane for it to be "the" lane
GUARD_CENTRE = (7, 10)    # own-frame tile columns always allowed


def threat_lane(state, team, sub, own):
    """0 / 1 (own-frame x half) holding the enemy push, or None."""
    hp = [0.0, 0.0]
    for e in state["entities"]:
        if e[1] == team or e[2] == 1:
            continue
        ox, oy = own(e[5], e[6], team, sub)
        if oy < 13_000:          # still deep on their side
            continue
        hp[0 if ox < 9_000 else 1] += max(0, e[7])
    tot = hp[0] + hp[1]
    if tot < GUARD_MIN_HP:
        return None
    for lane in (0, 1):
        if hp[lane] >= GUARD_SHARE * tot:
            return lane
    return None


def screen_tile(tx, ty):
    """Model tile (own frame, a half turn of the world) -> screen tile.

    The own frame is the replay convention: the board turned so we are at the
    bottom. The screen also puts us at the bottom, but as a MIRROR of that
    frame, not the same picture: the 2026-10-10 friendly (reader side 1)
    defended every left push in the right lane. Mirroring x fixes it; the
    derivation says the same holds for side 0, which is UNVERIFIED live.
    Every tile printed and logged here is a screen tile, x 0 = screen left.
    """
    return 17 - tx, ty


class Tapper:
    """One adb shell kept open; a play is two `input tap` lines written to it.

    A fresh `adb shell` per play cost a ~40 ms connection each time
    (measured 2026-10-10: round trip 39 ms, input tap 46-50 ms in total).
    """

    def __init__(self, serial):
        self.serial = serial
        self._open()

    def _open(self):
        self.p = subprocess.Popen([ADB, "-s", self.serial, "shell"], stdin=subprocess.PIPE,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

    def tap_pair(self, a, b):
        cmd = f"input tap {a[0]} {a[1]}; input tap {b[0]} {b[1]}\n"
        for _ in range(2):
            try:
                self.p.stdin.write(cmd)
                self.p.stdin.flush()
                return
            except (OSError, ValueError):
                self._open()

    def close(self):
        try:
            self.p.stdin.close()
            self.p.terminate()
        except (OSError, ValueError):
            pass


def other_bots():
    """PIDs of other live_play processes (a venv launcher and its child count as one run)."""
    import os
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*live_play*' } | "
         "ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId)\" }"],
        capture_output=True, text=True, timeout=30).stdout.split()
    pairs = [(int(out[i]), int(out[i + 1])) for i in range(0, len(out) - 1, 2)]
    mine = {os.getpid(), os.getppid()}
    return [p for p, parent in pairs if p not in mine and parent not in mine]


def probe_side(dev, default):
    """Which player record is ours: the one holding a real hand.

    Reads only the two hand-index vectors, to tell the records apart; neither
    is kept and the opponent's is never used. Our hand is 4 distinct deck
    indexes; the opponent's read as -1 in the first friendly. If both look
    valid the default is kept and a warning printed.
    """
    b = dev.battle()
    if not b:
        return default
    ps = dev.u64(b + 0xA8)
    valid = []
    for k in range(2):
        p = dev.u64(ps + 0xE0 + 8 * k)
        if not p:
            continue
        raw = dev.read(p, 0x220)
        side = struct.unpack_from("<i", raw, 0x78)[0]
        vec, size = struct.unpack_from("<Q", raw, 0x210)[0], struct.unpack_from("<i", raw, 0x21C)[0]
        if size != 4 or not vec:
            continue
        hand = struct.unpack("<4i", dev.read(vec, 16))
        if len(set(hand)) == 4 and all(0 <= h < 8 for h in hand):
            valid.append(side)
    if len(valid) == 1:
        return valid[0]
    print(f"  side probe ambiguous ({valid}), using side {default}")
    return default


def battle_result(frame, my_side):
    """(our crowns, their crowns, outcome) from the last battle frame.

    A destroyed princess tower leaves the entity list; a destroyed king stays
    with 0 hp and is 3 crowns (both seen in the 2026-10-10 friendly). Equal
    crowns: the lower lowest-tower hp loses (the game's tiebreak; UNVERIFIED
    that the reader's last frame shows the final hp), equal hp is a draw.
    """
    towers = {0: [], 1: []}
    for u in (frame or {}).get("u", []):
        name = u[10] if len(u) > 10 else ""
        if "Tower" in name and u[3] in towers:
            towers[u[3]].append((name, u[8]))

    def crowns_against(side):
        ts = towers[side]
        if any("King" in n and hp <= 0 for n, hp in ts):
            return 3
        return 2 - sum(1 for n, hp in ts if "Princess" in n and hp > 0)

    def lowest(side):
        return min((hp for _, hp in towers[side] if hp > 0), default=0)

    ours, theirs = crowns_against(1 - my_side), crowns_against(my_side)
    if ours != theirs:
        return ours, theirs, "win" if ours > theirs else "loss"
    if lowest(my_side) != lowest(1 - my_side):
        return ours, theirs, "win" if lowest(my_side) > lowest(1 - my_side) else "loss"
    return ours, theirs, "draw"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--deck", default="",
                    help="deck in the deck-screen order; overrides --deck-config")
    ap.add_argument("--deck-config", type=Path, default=DEFAULT_DECK_CONFIG)
    ap.add_argument("--side", type=int, default=1,
                    help="fallback when the side probe is ambiguous")
    ap.add_argument("--no-cap-rule", action="store_true")
    ap.add_argument("--no-queue", action="store_true", help="do not press Battle / OK")
    ap.add_argument("--no-lane-guard", action="store_true",
                    help="let Cannon / Ice Golem go to the lane without the enemy push")
    args = ap.parse_args()
    emulator.verify()
    others = other_bots()
    if others:
        print(f"refusing: another bot is already running (pids {others}). Two would both tap.")
        return 2
    stop_file = ROOT / "STOP"
    if stop_file.exists():
        print(f"refusing: {stop_file} exists. Delete it to run.")
        return 2

    import rs_rows as RR
    from rs_eval import DECIDE, RSAgent, _unit_names, to_world
    royalesim, names, ids, cat = RR._setup()
    sub = royalesim.SUBTILE_PER_MILLITILE
    unit_types = _unit_names()
    deck_config = load_deck_config(args.deck_config)
    deck_names = args.deck.split(",") if args.deck else list(deck_config.engine_cards)
    deck_forms = ([0] * 8 if args.deck else list(deck_config.forms))
    deck_rs = [ids[c] for c in deck_names]
    agent = RSAgent(args.ckpt)
    if getattr(agent.model, "ability_actions", 0):
        print("NOTE: the checkpoint can choose hero abilities offline, but live ability taps are "
              "disabled until the reader exposes and calibrates that button.")
    torch = agent.torch

    log_dir = ROOT / "reports" / "live_play"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / time.strftime("%Y%m%d_%H%M%S.jsonl")
    logf = log_path.open("a", encoding="utf-8")
    record_path = log_dir / "record.json"
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        record = {"win": 0, "loss": 0, "draw": 0}

    dev = GameMemory()
    side = args.side

    def start_reader(side_):
        p = subprocess.Popen(
            [ADB, "-s", dev.serial, "exec-out",
             f"HCR_SIDE={side_} /data/local/tmp/hcr_reader {dev.pid} {dev.libg:x} 25"],
            stdout=subprocess.PIPE, text=True)
        # A thread drains the pipe and keeps only the newest frame. Reading
        # line by line, a decision slower than the 50 ms frame interval left
        # a growing queue behind it, so the bot acted on ever older frames.
        p.latest, p.seq, p.cv = None, 0, threading.Condition()

        def pump():
            for line in p.stdout:
                with p.cv:
                    p.latest, p.seq = line, p.seq + 1
                    p.cv.notify()
            with p.cv:
                p.latest, p.seq = "", p.seq + 1
                p.cv.notify()
        threading.Thread(target=pump, daemon=True).start()
        return p

    def next_line(p, last_seq):
        with p.cv:
            p.cv.wait_for(lambda: p.seq != last_seq, timeout=5)
            return p.latest, p.seq

    proc = start_reader(side)
    tapper = Tapper(dev.serial)
    print(f"LIVE: tapping on {dev.serial}. Stop: Ctrl+C, or create {stop_file}.")
    print(f"Log: {log_path}   record so far: {record}")
    in_battle, last_frame, plays = False, None, 0
    ended_at, last_nav = None, 0.0
    nav = Navigator(dev.serial, queue=not args.no_queue)
    prev_tick_seen = None
    delays, misses = [], []
    prev_present = {0: set(), 1: set()}
    side_confirmed = False
    frozen_n, last_tick, last_end_tick = 0, -1, None
    seq = 0
    try:
        while True:
            line, seq_new = next_line(proc, seq)
            if seq_new == seq:
                continue                    # no frame for 5 s; keep waiting
            seq = seq_new
            if not line:
                print("reader exited; restarting it.")
                time.sleep(1)
                proc = start_reader(side)
                seq = 0
                continue
            if stop_file.exists():
                print("STOP file found, stopping.")
                break
            try:
                f = json.loads(line)
            except json.JSONDecodeError:
                print(f"reader said (not JSON): {line.rstrip()!r}")
                continue
            frozen = in_battle and f.get("t", -1) >= 0 and f.get("t") == last_tick
            frozen_n = frozen_n + 1 if frozen else 0
            last_tick = f.get("t", -1)
            # After a battle the client keeps its battle object (end screen) with
            # the clock stopped; those frames are not a new battle.
            stale = not in_battle and last_end_tick is not None and f.get("t") == last_end_tick
            # Out of a battle, a battle object whose clock is not moving (the
            # end screen, also when the bot is started on it) is not a battle.
            if not in_battle and not stale:
                stale = prev_tick_seen is None or f.get("t") == prev_tick_seen
            prev_tick_seen = f.get("t")
            if f.get("t", -1) < 0 or "me" not in f or frozen_n >= FROZEN_FRAMES or stale:
                if in_battle:
                    in_battle = False
                    ours, theirs, outcome = battle_result(last_frame, side)
                    if delays:
                        ds = sorted(delays)
                        print(f"    tap->board delay over {len(ds)} plays: median {ds[len(ds) // 2]} ticks, "
                              f"min {ds[0]}, max {ds[-1]} (look-ahead assumes {LEAD})")
                    if misses:
                        mx = sorted(m[0] for m in misses)[len(misses) // 2]
                        my = sorted(m[1] for m in misses)[len(misses) // 2]
                        print(f"    placement vs aim, median over {len(misses)}: x {mx:+.2f} y {my:+.2f} tiles "
                              f"(+y = toward the enemy)")
                    if plays == 0:
                        # Not a battle we played: a finished battle's object
                        # read at startup scored "win 0-0, 0 plays" (2026-10-10).
                        print("    battle ended with no plays from us; not counted")
                        ended_at = time.monotonic()
                        last_end_tick = f.get("t", -1)
                        continue
                    record[outcome] += 1
                    record_path.write_text(json.dumps(record))
                    n = sum(record.values())
                    logf.write(json.dumps({"result": outcome, "crowns": [ours, theirs],
                                           "plays": plays, "record": record}) + "\n")
                    logf.flush()
                    print(f"=== {outcome.upper()} {ours}-{theirs}   record W{record['win']} "
                          f"L{record['loss']} D{record['draw']}  "
                          f"({100 * record['win'] / max(1, n):.0f}% of {n})")
                    ended_at = time.monotonic()
                    last_end_tick = f.get("t", -1)
                if time.monotonic() - last_nav > 1.0:
                    last_nav = time.monotonic()
                    try:
                        note = nav.step()
                    except Exception as exc:          # a bad screenshot must not end the run
                        note = f"    navigation error {type(exc).__name__}: {exc}"
                    if note:
                        print(note)
                continue
            if not in_battle:
                # New battle: find our side, reset everything kept per battle.
                new_side = probe_side(dev, side)
                if new_side != side:
                    side = new_side
                    proc.terminate()
                    proc = start_reader(side)
                    seq = 0
                    print(f"--- our side is {side}; reader restarted")
                    continue
                in_battle = True
                ended_at = None
                print(f"--- battle started, our side {side}")
                agent.lead, agent.lead_rebuild, agent.sent = LEAD, True, []
                mem = {"seen": {0: set(), 1: set()}, "last": {0: [], 1: []}}
                known, last_opp_play = set(), {}
                plays, last_decide = 0, -10**9
                pending = []
                awaiting = []
                prev_present = {0: set(), 1: set()}
                last_rebuild_error = None
                side_confirmed = False
            last_frame = f
            now = f["t"]
            # Side check from the board: a unit of a card that is not in our
            # deck is the opponent's. If one stands on "our" side, the side is
            # wrong - switch before another decision is made on their hand.
            if not side_confirmed:
                for u in f["u"]:
                    name = u[10] if len(u) > 10 else ""
                    if "Tower" in name or "Projectile" in name or "Container" in name:
                        continue
                    entry = CARD_MAP.get(str(u[6]))
                    c = entry["rs_id"] if entry else card_of_form(name, cat)
                    if c >= 0 and c not in deck_rs:
                        side_confirmed = True
                        if u[3] == side:
                            side = 1 - side
                            print(f"--- {cat[c][0]} is on side {u[3]}: we are side {side}; reader restarted")
                            proc.terminate()
                            proc = start_reader(side)
                            seq = 0
                            in_battle = False
                        break
                if not in_battle:
                    continue
            state = rs_state_from_frame(f, side, deck_rs, unit_types, cat, sub, deck_forms)
            team = next((t for t in (0, 1) if state["players"][t]["hand"]), None)
            if team is None:
                continue
            # Opponent plays, from the raw frame so spells count too (they are
            # not units). One play = the first new entity of a card; more of
            # the same card within PLAY_WINDOW ticks (3 Skeletons, a Log and
            # its roller, Ice Spirits and the freeze) are the same play.
            # Which card each object belongs to, and which cards were on the
            # board last frame. A new object is a PLAY only if its card had
            # nothing on the board in the previous frame, and only once per
            # frame. Shots (TowerCannonball, MinionSpit, ArcherArrow), spawns
            # (a Witch's or a Graveyard's skeletons) and explosions all appear
            # while their source is alive; counted as plays, they filled the
            # model's "opponent's last 4 plays" input with junk in exactly the
            # busy fights where it then played badly (2026-10-10 log: 11
            # "Cannon" plays that were cannonballs, 10 "Minions" that were spit).
            def card_of(u):
                entry = CARD_MAP.get(str(u[6]))
                return entry["rs_id"] if entry else card_of_form(u[10] if len(u) > 10 else "", cat)
            present = {0: set(), 1: set()}
            counted = set()
            for u in f["u"]:
                if u[3] in present:
                    present[u[3]].add(card_of(u))
            for u in f["u"]:
                addr, uside, x, y, card_native = u[0], u[3], u[4], u[5], u[6]
                if addr in known:
                    continue
                c_here = card_of(u)
                fresh = (c_here >= 0 and c_here not in prev_present.get(uside, set())
                         and (uside, c_here) not in counted)
                if not fresh:
                    known.add(addr)
                    continue
                counted.add((uside, c_here))
                if uside == side and addr not in known:
                    # Our own play appearing: tap -> on-board delay, measured
                    # per play. The look-ahead assumes LEAD ticks; this says
                    # what it really is on this rig.
                    known.add(addr)
                    entry = CARD_MAP.get(str(card_native))
                    name = u[10] if len(u) > 10 else ""
                    c = entry["rs_id"] if entry else card_of_form(name, cat)
                    for k, (t_tap, c_tap, w_tap, aim, tm) in enumerate(awaiting):
                        if c_tap == c:
                            del awaiting[k]
                            lag = now - t_tap
                            delays.append(lag)
                            # Where it landed against where it was aimed, in
                            # tiles of our own frame (+y = toward the enemy).
                            # Settles whether "too far" is the screen mapping
                            # or the model's choice.
                            ax, ay = RR.own(aim[0], aim[1], tm, sub)
                            lx, ly = RR.own(x * sub, y * sub, tm, sub)
                            dx, dy = (lx - ax) / 1000, (ay - ly) / 1000
                            misses.append((dx, dy))
                            logf.write(json.dumps({"t": now, "landed": cat[c][0], "delay_ticks": lag,
                                                   "wall_ms": round(1000 * (time.monotonic() - w_tap)),
                                                   "miss_tiles": [round(dx, 2), round(dy, 2)]}) + "\n")
                            print(f"t={now:5d}   landed {cat[c][0]} {lag} ticks after the tap decision "
                                  f"({round(1000 * (time.monotonic() - w_tap))} ms wall), "
                                  f"off aim by x {dx:+.1f} y {dy:+.1f} tiles")
                            break
                    continue
                if uside == side or addr in known:
                    continue
                known.add(addr)
                entry = CARD_MAP.get(str(card_native))
                name = u[10] if len(u) > 10 else ""
                card = entry["rs_id"] if entry else card_of_form(name, cat)
                # A Wall Breaker's projectile, a Skeleton Barrel's container, a
                # Blowdart Goblin's dart carry the card id too: not new plays.
                if "Projectile" in name or "Container" in name:
                    continue
                if card < 0 or now - last_opp_play.get(card, -10**9) < PLAY_WINDOW:
                    continue
                last_opp_play[card] = now
                mem["seen"][1 - team].add(card + 2)
                mem["last"][1 - team].append((card + 2, now))
                ox, oy = RR.own(x * sub, y * sub, team, sub)
                sx, sy = screen_tile(int(ox // 1000), int(oy // 1000))
                rec = {"t": now, "opp": cat[card][0], "unit": name, "tile": [sx, sy]}
                logf.write(json.dumps(rec) + "\n")
                print(f"t={now:5d}   opp  {rec['opp']} ({name or 'spell'}) tile ({sx},{sy})")
            prev_present = present
            # Opponent deck for the rebuild: cards seen so far, padded. It makes
            # no plays in the look-ahead, so the padding never acts.
            opp_seen = [c - 2 for c in sorted(mem["seen"][1 - team])]
            state["players"][1 - team]["deck"] = (opp_seen + [ids[n] for n in FILLER
                                                             if ids[n] not in opp_seen])[:8]
            state["players"][1 - team]["forms"] = [0] * len(state["players"][1 - team]["deck"])
            p = state["players"][team]
            # Plays sent but not yet shown by the reader (~1.1 s). Instead of
            # freezing until they land, their slot is blocked and their cost
            # is taken off the elixir we can spend, so a second card can
            # follow at once (a two-card defence) without double-playing.
            pending = [(sl, c, cost, t0) for (sl, c, cost, t0) in pending
                       if sl < len(p["hand"]) and p["hand"][sl] == c and now - t0 < PENDING_TICKS]
            spendable = p["elixir_milli"] - 1000 * sum(cost for _, _, cost, _ in pending)
            blocked = {sl for sl, _, _, _ in pending}
            if now - last_decide < DECIDE_LIVE:
                continue
            dt = min(DECIDE, max(1, now - last_decide))
            last_decide = now
            t_decide = time.perf_counter()
            agent._team = team
            view = agent.ahead(None, state)
            row = RR.obs_row(view, team, sub, mem, now + agent.lead, live=agent.live)
            b = agent.batch(row)
            with torch.no_grad():
                gate, _, _ = agent.model(b, card_ids=b["hand"][:, 0].clamp(min=0))
            g = gate[0].float().numpy()
            g[0] += agent.wait_bias
            # See the startup note: never guess an uncalibrated paid tap.
            if len(g) > 5:
                g[5:] = -1e9
            for slot in range(4):
                if (slot >= len(p["hand"]) or slot in blocked
                        or spendable < p["hand_costs"][slot] * 1000):
                    g[1 + slot] = -1e9
            if g[1:].max() < -1e8:
                continue
            pr = np.exp(g - g.max())
            pr /= pr.sum()
            capped = not args.no_cap_rule and spendable >= CAP_ELIXIR
            # Same wait rate per second as the 0.5 s evaluator cadence, at any cadence.
            if not capped and agent.rng.random() < float(pr[0]) ** (dt / 20):
                continue
            slot = int(np.argmax(g[1:]))
            with torch.no_grad():
                _, tile, _ = agent.model(b, card_ids=b["hand"][:, slot])
            order = np.argsort(-tile[0].float().numpy())
            check = agent.rebuilt(state, team, royalesim)
            if check is None:
                why = getattr(agent, "rebuild_error", "?")
                if why != last_rebuild_error:
                    print(f"t={now} rebuild failed ({why}); skipping decisions until it works")
                    last_rebuild_error = why
                continue
            target = None
            lane = (threat_lane(state, team, sub, RR.own)
                    if cat[p["hand"][slot]][0] in GUARD_CARDS and not args.no_lane_guard else None)
            for t in order[:60]:
                tx, ty = int(t % 18), int(t // 18)
                if (lane is not None and not GUARD_CENTRE[0] <= tx <= GUARD_CENTRE[1]
                        and (0 if tx < 9 else 1) != lane):
                    continue
                wx, wy = to_world(tx * 1000 + 500, ty * 1000 + 500, team)
                if check.check_deploy(team, slot, wx * sub, wy * sub) == 0:
                    target = (tx, ty, wx, wy)
                    break
            if target is None:
                continue
            tx, ty, wx, wy = target
            card = p["hand"][slot]
            sx, sy = screen_tile(tx, ty)
            px = (round(_X0 + _XS * sx), round(_Y0 + _YS * sy))
            tapper.tap_pair(CARD_CENTRES[slot], px)
            plays += 1
            agent.sent.append((now + LEAD, card, wx * sub, wy * sub))
            pending.append((slot, card, p["hand_costs"][slot], now))
            awaiting.append((now, card, time.monotonic(), (wx * sub, wy * sub), team))
            rec = {"t": now, "card": cat[card][0], "slot": slot, "tile": [sx, sy],
                   "model_tile": [tx, ty], "px": px, "elixir": p["elixir_milli"],
                   "p_wait": float(pr[0]), "capped": capped,
                   "hand": [cat[c][0] for c in p["hand"]],
                   "ms": round((time.perf_counter() - t_decide) * 1000)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"t={now:5d} PLAY #{plays} {rec['card']} tile ({sx},{sy}) elixir "
                  f"{p['elixir_milli'] / 1000:.1f}{' (cap rule)' if capped else ''}  [{rec['ms']} ms]")
    except KeyboardInterrupt:
        print("Ctrl+C, stopping.")
    finally:
        proc.terminate()
        tapper.close()
        logf.close()
    print(f"record: {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
