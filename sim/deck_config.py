"""Validated, reusable deck configuration for replay training and RoyaleSim.

The replay corpus names cards with RoyaleAPI slugs (``giant-snowball-ev1``),
while RoyaleSim uses catalogue names (``Snowball``) plus a parallel numeric
form list.  Keeping both spellings in one small JSON file prevents the row
builder, evaluator, self-play loop and live bridge from silently selecting
different decks.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECK_CONFIG = ROOT / "configs" / "decks" / "hog_26.json"
FORM_CODE = {"base": 0, "evolution": 1, "hero": 2, "any": -1}
FORM_SUFFIX = {0: "", 1: "-ev1", 2: "-hero"}


def split_replay_form(key: str) -> tuple[str, int]:
    """Return the base RoyaleAPI key and form code for a replay card key."""
    if key.endswith("-hero"):
        return key[:-5], 2
    if "-ev" in key and key.rsplit("-ev", 1)[1].isdigit():
        return key.rsplit("-ev", 1)[0], 1
    return key, 0


@dataclass(frozen=True)
class DeckCard:
    engine: str
    replay: str
    form: int = 0

    @property
    def replay_key(self) -> str:
        return self.replay + FORM_SUFFIX.get(self.form, "")


@dataclass(frozen=True)
class DeckConfig:
    name: str
    cards: tuple[DeckCard, ...]
    match_forms: bool = True
    description: str = ""
    source: Path | None = None

    @property
    def engine_cards(self) -> tuple[str, ...]:
        return tuple(card.engine for card in self.cards)

    @property
    def forms(self) -> tuple[int, ...]:
        return tuple(max(0, card.form) for card in self.cards)

    @property
    def replay_keys(self) -> frozenset[str]:
        return frozenset(card.replay_key for card in self.cards)

    @property
    def replay_bases(self) -> frozenset[str]:
        return frozenset(card.replay for card in self.cards)

    @property
    def label(self) -> str:
        return self.name.replace("_", " ")

    def matches_replay_cards(self, cards: Sequence[Mapping[str, object]]) -> bool:
        keys = [str(card.get("card_key") or "") for card in cards]
        if len(keys) != 8 or len(set(keys)) != 8:
            return False
        if self.match_forms:
            return frozenset(keys) == self.replay_keys
        return frozenset(split_replay_form(key)[0] for key in keys) == self.replay_bases

    def target_seat(self, battle: Mapping[str, object]) -> int | None:
        """0 for replay ``team``, 1 for ``opponent``, else ``None``.

        A mirror is intentionally rejected.  Evaluation and self-play need one
        unambiguous seat for the target policy.
        """
        found: list[int] = []
        for seat, side in enumerate(("team", "opponent")):
            try:
                players = battle[side]["players"]  # type: ignore[index]
                if len(players) == 1 and self.matches_replay_cards(players[0]["deck"]):
                    found.append(seat)
            except (KeyError, IndexError, TypeError):
                continue
        return found[0] if len(found) == 1 else None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "match_forms": self.match_forms,
            "engine_cards": list(self.engine_cards),
            "forms": list(self.forms),
            "replay_keys": sorted(self.replay_keys),
            "source": str(self.source) if self.source else "",
        }


def load_deck_config(path: str | Path | None = None) -> DeckConfig:
    source = Path(path) if path else DEFAULT_DECK_CONFIG
    if not source.is_absolute():
        source = ROOT / source
    raw = json.loads(source.read_text(encoding="utf-8"))
    cards = []
    for i, item in enumerate(raw.get("cards", [])):
        if not isinstance(item, dict):
            raise ValueError(f"{source}: cards[{i}] must be an object")
        form_name = str(item.get("form", "base")).lower()
        if form_name not in FORM_CODE:
            raise ValueError(f"{source}: cards[{i}].form must be one of {sorted(FORM_CODE)}")
        cards.append(DeckCard(engine=str(item.get("engine", "")).strip(),
                              replay=str(item.get("replay", "")).strip(),
                              form=FORM_CODE[form_name]))
    cfg = DeckConfig(name=str(raw.get("name") or source.stem), cards=tuple(cards),
                     match_forms=bool(raw.get("match_forms", True)),
                     description=str(raw.get("description") or ""), source=source.resolve())
    validate_deck_config(cfg)
    return cfg


def validate_deck_config(cfg: DeckConfig) -> None:
    if len(cfg.cards) != 8:
        raise ValueError(f"{cfg.source or cfg.name}: a deck must contain exactly 8 cards")
    if any(not card.engine or not card.replay for card in cfg.cards):
        raise ValueError(f"{cfg.source or cfg.name}: every card needs engine and replay names")
    if len(set(cfg.engine_cards)) != 8 or len(cfg.replay_bases) != 8:
        raise ValueError(f"{cfg.source or cfg.name}: duplicate cards are not allowed")
    if sum(card.form == 1 for card in cfg.cards) > 2:
        raise ValueError(f"{cfg.source or cfg.name}: at most two evolution slots are allowed")
    if sum(card.form == 2 for card in cfg.cards) > 1:
        raise ValueError(f"{cfg.source or cfg.name}: at most one hero slot is allowed")
    if cfg.match_forms and any(card.form < 0 for card in cfg.cards):
        raise ValueError(f"{cfg.source or cfg.name}: form 'any' requires match_forms=false")


def ability_buttons(deck: Sequence[str], forms: Sequence[int], catalogue: Sequence[Sequence]) -> list[str]:
    """RoyaleSim ability button card names, in engine button order."""
    by_name = {str(row[0]): row for row in catalogue}
    heroes = [name for name, form in zip(deck, forms) if form == 2]
    champions = [name for name in deck
                 if len(by_name.get(name, ())) > 11 and bool(by_name[name][11])]
    return heroes + [name for name in champions if name not in heroes]


def ability_button_for_event(event: Mapping[str, object], buttons: Sequence[str],
                             replay_names: Mapping[str, str]) -> int | None:
    """Resolve an IL_Replay ability event to its RoyaleSim button index."""
    raw = event.get("ability_source_candidates") or []
    if isinstance(raw, str):
        raw = [raw]
    candidates = []
    for key in raw if isinstance(raw, Iterable) else ():
        base, _ = split_replay_form(str(key))
        name = replay_names.get(base)
        if name in buttons and name not in candidates:
            candidates.append(name)
    if len(candidates) != 1:
        return None
    return buttons.index(candidates[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate and display a deck config")
    ap.add_argument("config", nargs="?", default=str(DEFAULT_DECK_CONFIG))
    args = ap.parse_args()
    cfg = load_deck_config(args.config)
    print(json.dumps(cfg.metadata(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
