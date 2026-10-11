from pathlib import Path

from sim.deck_config import (ability_button_for_event, ability_buttons,
                             load_deck_config)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "decks" / "goblin_drill_evo_hero.json"


def replay_deck(keys):
    return [{"card_key": key} for key in keys]


def battle(team, opponent):
    return {"team": {"players": [{"deck": replay_deck(team)}]},
            "opponent": {"players": [{"deck": replay_deck(opponent)}]}}


def test_exact_forms_and_target_seat():
    cfg = load_deck_config(CONFIG)
    exact = list(cfg.replay_keys)
    other = ["hog-rider", "musketeer", "cannon", "fireball", "skeletons",
             "ice-spirit", "ice-golem", "the-log"]
    assert cfg.matches_replay_cards(replay_deck(exact))
    assert cfg.target_seat(battle(exact, other)) == 0
    assert cfg.target_seat(battle(other, exact)) == 1


def test_wrong_evolution_or_hero_is_rejected():
    cfg = load_deck_config(CONFIG)
    wrong_drill = [key.replace("goblin-drill-ev1", "goblin-drill")
                   for key in cfg.replay_keys]
    wrong_knight = [key.replace("knight-hero", "knight") for key in cfg.replay_keys]
    assert not cfg.matches_replay_cards(replay_deck(wrong_drill))
    assert not cfg.matches_replay_cards(replay_deck(wrong_knight))


def test_mirror_is_ambiguous_and_rejected():
    cfg = load_deck_config(CONFIG)
    exact = list(cfg.replay_keys)
    assert cfg.target_seat(battle(exact, exact)) is None


def test_hero_ability_event_maps_to_button():
    cfg = load_deck_config(CONFIG)
    catalogue = [[name] + [None] * 11 for name in cfg.engine_cards]
    buttons = ability_buttons(cfg.engine_cards, cfg.forms, catalogue)
    assert buttons == ["Knight"]
    event = {"ability_source_candidates": ["knight-hero"]}
    replay_names = {card.replay: card.engine for card in cfg.cards}
    assert ability_button_for_event(event, buttons, replay_names) == 0
