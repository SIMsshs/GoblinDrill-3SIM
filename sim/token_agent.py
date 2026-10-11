"""A trained `TokenPolicy` playing a live `Match`, with the same observation
the training rows were built from (`sim.replay_rows`).

The agent sees only what a player sees: the board, its own hand and elixir,
and the opponent's plays as they happen. Opponent plays are noticed from the
public consequence - the opponent's spent elixir rising and its last card
changing - never from its hand.

Two corrections between training and play:

* Wait rows were subsampled to `wait_keep` when training, so the model's
  wait logit is low by log(1 / wait_keep). It is added back here.
* Unaffordable cards and illegal tiles are masked. The model was trained on
  real players, who never tap a card they cannot afford.
* Play-or-wait is *sampled*, not argmaxed. A player plays roughly once every
  five seconds, so at any one decision "wait" is the most likely action and an
  argmax waits forever (0.8 plays a match in the first test). The wait
  probability was learned per 1 s row and the agent decides every
  `decide_ms`, so it is rescaled: p_wait ** (decide_ms / 1000). Which card and
  which tile are still the model's best.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch

from .arena import Point, deploy_area_ok, in_arena
from .replay_rows import (MAX_UNITS, PAD, UNIT_FEATURES, UNK, _orient, _towers,
                          card_vocab, cycle_hand, unit_vocab)

GRID_W, GRID_H = 18, 32
# Match card names (the bot's) -> engine table names (the vocabulary's).
ALIASES = {"ice_golem": "ice_golemite", "ice_spirit": "ice_spirits", "the_log": "log"}


def vocab_name(card: str) -> str:
    return ALIASES.get(card, card)


class TokenAgent:
    def __init__(self, checkpoint, side: int, device: str = "cpu",
                 wait_keep: float = 0.35, sample: bool = False, seed: int = 0,
                 decide_ms: int = 500):
        from .model_tokens import TokenPolicy
        blob = torch.load(checkpoint, map_location=device)
        meta = blob["meta"]
        self.model = TokenPolicy(meta["n_cards"], meta["n_units"], d=meta.get("d", 192),
                                 layers=meta.get("layers", 4), heads=meta.get("heads", 6),
                                 ability_actions=meta.get("ability_actions", 0),
                                 ability_features=meta.get("ability_features", 4)).to(device).eval()
        self.model.load_state_dict(blob["model"])
        self.side = side
        self.device = device
        self.wait_bias = math.log(1.0 / wait_keep)
        self.sample = sample
        self.decide_ms = decide_ms
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.history: List[str] = []
        self.opp_seen: set = set()
        self.opp_last: List[tuple] = []
        self._opp_spent = 0
        self.plays: Dict[str, int] = {}

    # ---------------------------------------------------------- observation
    def _watch_opponent(self, match) -> None:
        opp = match.players[-self.side]
        if opp.elixir_spent != self._opp_spent and opp.last_card:
            cid = card_vocab().get(vocab_name(opp.last_card), UNK)
            self.opp_seen.add(cid)
            self.opp_last.append((cid, match.elapsed_ms // 50))
        self._opp_spent = opp.elixir_spent

    def _obs(self, match):
        side = self.side
        player = match.players[side]
        tick = match.elapsed_ms // 50
        vocab_u = unit_vocab()
        king = match.towers[side]["king"].pos
        live = [e for e in match.battle.entities.values()
                if getattr(e, "alive", True) and not e.is_tower]
        live.sort(key=lambda e: (e.pos.x - king.x) ** 2 + (e.pos.y - king.y) ** 2)
        ids = np.zeros(MAX_UNITS, dtype=np.int64)
        feats = np.zeros((MAX_UNITS, UNIT_FEATURES), dtype=np.float32)
        for k, e in enumerate(live[:MAX_UNITS]):
            ox, oy = _orient(e.pos.x, e.pos.y, side)
            ids[k] = vocab_u.get(e.name, UNK)
            feats[k] = (1.0 if e.side == side else 0.0, ox / 18_000, oy / 32_000,
                        e.hitpoints / max(1, e.max_hitpoints),
                        1.0 if e.flying else 0.0, 1.0 if e.is_building else 0.0,
                        1.0 if getattr(e, "deploy_remaining_ms", 0) > 0 else 0.0)
        deck = [vocab_name(c) for c in player.deck]
        real = cycle_hand(deck, self.history)
        # The engine's own hand is exact in play (no forcing happens here);
        # cycle_hand is only a cross-check, so the engine hand is used.
        hand_names = [vocab_name(c) for c in player.hand[:4]]
        cv = card_vocab()
        hand = [cv.get(c, UNK) for c in hand_names] + [PAD] * (4 - len(hand_names))
        t_ms = match.elapsed_ms
        seen = sorted(self.opp_seen)[:8]
        last = self.opp_last[-4:]
        b = {
            "unit_ids": ids[None], "unit_feats": feats[None],
            "towers": _towers(match, side)[None],
            "scalars": np.asarray([[player.elixir / 10_000, min(1.0, t_ms / 300_000),
                                    1.0 if t_ms >= 120_000 else 0.0,
                                    1.0 if t_ms >= 180_000 else 0.0]], dtype=np.float32),
            "hand": np.asarray([hand], dtype=np.int64),
            "next": np.asarray([cv.get(vocab_name(player.next_card or ""), PAD)
                                if player.next_card else PAD], dtype=np.int64),
            "opp_seen": np.asarray([seen + [PAD] * (8 - len(seen))], dtype=np.int64),
            "opp_last": np.asarray([[c for c, _ in last] + [PAD] * (4 - len(last))], dtype=np.int64),
            "opp_last_age": np.asarray([[min(1.0, (tick - t) / 400) for _, t in last]
                                        + [1.0] * (4 - len(last))], dtype=np.float32),
        }
        return {k: torch.from_numpy(v).to(self.device) for k, v in b.items()}, real

    # --------------------------------------------------------------- acting
    def _legal_tiles(self, match, card: str) -> np.ndarray:
        side = self.side
        spec = match.cards.get(card)
        spell = spec is not None and spec.unit is None
        down = [lane for lane, t in match.towers[-side].items()
                if lane != "king" and not t.alive]
        mask = np.zeros(GRID_W * GRID_H, dtype=bool)
        for ty in range(GRID_H):
            for tx in range(GRID_W):
                # Tile in the player's own frame -> world point at the tile centre.
                ox, oy = tx * 1000 + 500, ty * 1000 + 500
                wx, wy = _orient(ox, oy, side)
                p = Point(wx, wy)
                mask[ty * GRID_W + tx] = in_arena(p) if spell else deploy_area_ok(p, side, down)
        return mask

    @torch.no_grad()
    def act(self, match) -> Optional[tuple]:
        self._watch_opponent(match)
        b, _ = self._obs(match)
        player = match.players[self.side]
        gate, _, _ = self.model(b, card_ids=b["hand"][:, 0].clamp(min=0))
        gate = gate[0].float().cpu().numpy()
        gate[0] += self.wait_bias
        # This adapter drives HastyCR's older Match API, not RoyaleSim's
        # explicit ability-button slots. The RoyaleSim agent handles them.
        if len(gate) > 5:
            gate[5:] = -1e9
        for slot, card in enumerate(player.hand[:4]):
            if not match.can_play(self.side, card):
                gate[1 + slot] = -1e9
        for slot in range(len(player.hand), 4):
            gate[1 + slot] = -1e9
        p = np.exp(gate - gate.max())
        p /= p.sum()
        p_wait = float(p[0]) ** (self.decide_ms / 1000.0)
        if self.rng.random() < p_wait or not np.isfinite(gate[1:5]).any() or gate[1:5].max() < -1e8:
            return None
        slots = gate[1:5]
        if self.sample:
            q = np.exp(slots - slots.max())
            choice = 1 + int(self.rng.choice(4, p=q / q.sum()))
        else:
            choice = 1 + int(slots.argmax())
        if choice == 0:
            return None
        card = player.hand[choice - 1]
        card_id = b["hand"][:, choice - 1]
        _, tile, _ = self.model(b, card_ids=card_id)
        tile = tile[0].float().cpu().numpy()
        legal = self._legal_tiles(match, card)
        if not legal.any():
            return None
        tile[~legal] = -1e9
        t = int(tile.argmax())
        tx, ty = t % GRID_W, t // GRID_W
        wx, wy = _orient(tx * 1000 + 500, ty * 1000 + 500, self.side)
        if not match.play_card(self.side, card, Point(wx, wy)):
            return None
        self.history.append(vocab_name(card))
        self.plays[card] = self.plays.get(card, 0) + 1
        return card, tx, ty, "token"
