"""A shadow battle: the live match so far, re-driven in RoyaleSim from public
information only.

Search and delay compensation both need a copy of the battle to run forward.
The true engine state is not available live, and a battle rebuilt from what
is on screen (`RSAgent.rebuilt`) loses everything in flight - it measured
worse than no compensation at all. The shadow keeps all of that because it is
not rebuilt: it is the whole match replayed from tick 0 with every play seen
so far, which RoyaleSim does in a few tens of milliseconds.

Public events only:
* our own plays - exact: we sent them (card, tile, tick they execute);
* the opponent's troop and building plays - a new opponent unit whose unit
  type is the card's own unit (so a Witch's skeletons or a Golem's Golemites
  are not mistaken for new plays), grouped by card per observation, placed at
  the group's centroid, timed at the observation tick;
* the opponent's spells - a new opponent spell object of a SPELL-kind card,
  aimed where it is aimed, cast `ticks_flown` ticks ago.
The opponent's deck is the cards seen so far, padded with cards never played;
both decks are dealt in an order that explains every play (as the replay
re-drive does). Levels: our own exact, the opponent's from the level its units
show, towers from the match level.

`Shadow.battle(now)` returns a fresh Battle at tick `now`; callers save/load
it freely. Re-driving happens only when a new event arrives; otherwise the
cached shadow is stepped forward.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from royalesim_redrive import deal_order

Event = Tuple[int, int, int, int, int]      # (tick, team, card_id, x_sub, y_sub)
DEPLOY_TICKS = 20                           # the usual troop deploy time, 1 s (assumption)


class Shadow:
    def __init__(self, royalesim, catalogue, me: int, my_deck: List[int], my_levels: List[int],
                 tower_level: int, prim: Dict[int, int], my_forms: Optional[List[int]] = None):
        self.rs = royalesim
        self.cat = catalogue
        self.me, self.opp = me, 1 - me
        self.my_deck, self.my_levels = list(my_deck), list(my_levels)
        self.my_forms = list(my_forms or [0] * len(my_deck))
        self.tower_level = tower_level
        self.prim = prim
        self.events: List[Event] = []
        self.seen_uids: set = set()
        self.seen_spells: set = set()
        self.opp_cards: Dict[int, int] = {}            # card -> level seen
        self._battle = None
        self._tick = 0
        self._dirty = True
        self.refused = 0
        self.why: Dict[tuple, int] = {}         # (team, reason) -> count, last rebuild
        self.F = {k: i for i, k in enumerate(royalesim.ENTITY_FIELDS)}
        self.unit_names = json.loads(royalesim.Battle(card_names=None, slot_of_k=[[0, 1, 2], [0, 1, 2]])
                                     .unit_types_json())
        self.SF = {k: i for i, k in enumerate(royalesim.SPELL_FIELDS)}

    # ------------------------------------------------------------- events
    def own_play(self, tick_exec: int, card: int, x_sub: int, y_sub: int) -> None:
        self.events.append((tick_exec, self.me, card, x_sub, y_sub))
        self._dirty = True

    def observe(self, state: dict) -> None:
        F, SF = self.F, self.SF
        tick = state["tick"]
        groups: Dict[int, list] = {}
        for e in state["entities"]:
            uid = e[F["uid"]]
            if uid in self.seen_uids:
                continue
            self.seen_uids.add(uid)
            if e[F["team"]] != self.opp or e[F["tower_slot"]] >= 0:
                continue
            card = e[F["card_id"]]
            if self.prim.get(card) != e[F["unit_type"]]:
                # Not the card's own unit type: a child (a Witch's skeleton) -
                # unless it is an evolved or hero form of that unit
                # (Skeleton_EV1, MusketeerHero), which is the play itself.
                # Missing that dropped every evolved / hero play by the
                # opponent from the shadow.
                base = self.unit_names[self.prim[card]].lower() if card in self.prim else ""
                if not (base and self.unit_names[e[F["unit_type"]]].lower().startswith(base)):
                    continue
            groups.setdefault(card, []).append(e)
            self.opp_cards[card] = e[F["level"]] or self.opp_cards.get(card, 0)
        sub = self.rs.SUBTILE_PER_MILLITILE
        for card, units in groups.items():
            x = sum(u[F["x"]] for u in units) // len(units)
            y = sum(u[F["y"]] for u in units) // len(units)
            # Back-date to the tap: a unit appears with a deploy timer that has
            # been running since it was placed. Most troops deploy in 20 ticks
            # (1 s); the remaining count says how much of it has passed.
            # (DEPLOY_TICKS is an assumption for cards whose deploy differs.)
            remaining = max(u[F["deploy_ticks"]] for u in units)
            played = tick - max(0, DEPLOY_TICKS - remaining) if remaining > 0 else tick - 10
            played = max(0, min(tick, played))
            # A card whose units appear over several ticks (staggered deploys)
            # is one play, not several: merge a sighting of the same card
            # within 2 s and 3 tiles of the last one.
            dup = any(t2 >= tick - 40 and team == self.opp and c2 == card
                      and abs(x2 - x) <= 3000 * sub and abs(y2 - y) <= 3000 * sub
                      for (t2, team, c2, x2, y2) in self.events[-12:])
            if dup:
                continue
            self.events.append((played, self.opp, card, x, y))
            self._dirty = True
        for sp in state.get("spells", []):
            if sp[SF["team"]] != self.opp:
                continue
            card = sp[SF["card_id"]]
            if card < 0 or self.cat[card][8] != "SPELL":
                continue
            cast = tick - int(sp[SF["ticks_flown"]] or 0)
            key = (card, cast // 5)
            if key in self.seen_spells:
                continue
            self.seen_spells.add(key)
            self.opp_cards.setdefault(card, 0)
            self.events.append((cast, self.opp, card, sp[SF["aim_x"]], sp[SF["aim_y"]]))
            self._dirty = True

    # ------------------------------------------------------------ battle
    def _opp_deck(self) -> List[int]:
        seen = list(self.opp_cards)[:8]
        fill = [i for i, row in enumerate(self.cat)
                if i not in seen and i not in self.my_deck and row[8] != "SPELL"]
        return seen + fill[:8 - len(seen)]

    def _rebuild(self, now: int):
        evs = sorted(self.events)
        decks = {self.me: self.my_deck, self.opp: self._opp_deck()}
        dealt = {}
        for team in (self.me, self.opp):
            order = deal_order(decks[team], [c for (_, t, c, _, _) in evs if t == team])
            dealt[team] = order or decks[team]
        lv = {self.me: dict(zip(self.my_deck, self.my_levels)),
              self.opp: {c: (self.opp_cards.get(c) or self.tower_level) for c in decks[self.opp]}}
        form = {self.me: dict(zip(self.my_deck, self.my_forms)),
                self.opp: {c: 0 for c in decks[self.opp]}}
        b = self.rs.Battle(card_names=None, slot_of_k=[[0, 1, 2], [0, 1, 2]])
        b.reset(seed=0, decks=[dealt[0], dealt[1]], shuffle=0, start_tick=0,
                elixir_milli=[6000, 6000], tower_hp=None, spawns=[],
                levels=[[lv[t][c] or self.tower_level for c in dealt[t]] for t in (0, 1)],
                forms=[[form[t].get(c, 0) for c in dealt[t]] for t in (0, 1)],
                tower_levels=[self.tower_level, self.tower_level])
        tick = 0
        self.why = {}
        for (t, team, card, x, y) in evs:
            if t > now:
                break
            if t > tick:
                b.step([], t - tick)
                tick = t
            st = json.loads(b.state_json())
            hand = st["players"][team]["hand"]
            if card not in hand:
                self.refused += 1
                k = (team, "not in hand")
                self.why[k] = self.why.get(k, 0) + 1
                continue
            slot = hand.index(card)
            if b.check_deploy(team, slot, x, y) != 0:
                # Seen after it moved: walk the tap back toward its own side
                # until the engine accepts it.
                sub = self.rs.SUBTILE_PER_MILLITILE
                step_y = (-500 if team == 0 else 500) * sub
                for k in range(1, 9):
                    if b.check_deploy(team, slot, x, y + k * step_y) == 0:
                        y = y + k * step_y
                        break
            r = b.step([(team, slot, x, y)], 1)
            tick += 1
            if r and self.rs.DEPLOY_REASONS[r[0][1]] != "OK":
                self.refused += 1
                k = (team, self.rs.DEPLOY_REASONS[r[0][1]], self.cat[card][0])
                self.why[k] = self.why.get(k, 0) + 1
        if now > tick:
            b.step([], now - tick)
        return b, now

    def battle(self, now: int):
        if self._dirty or self._battle is None or now < self._tick:
            self._battle, self._tick = self._rebuild(now)
            self._dirty = False
        elif now > self._tick:
            self._battle.step([], now - self._tick)
            self._tick = now
        return self._battle

    def pending_own(self, now: int) -> List[Event]:
        """Our own plays sent but not yet executed at `now`."""
        return [e for e in self.events if e[1] == self.me and e[0] > now]
