import numpy as np
import pytest

from tools.rs_eval import next_wakeup_tick, unsupported_deck_cards
from tools.rs_selfplay import evaluation_score, gate_probs, linear_schedule, shaped_return


def test_curriculum_schedule_reaches_both_endpoints():
    assert linear_schedule(0.25, 0.75, 0, 20) == 0.25
    assert linear_schedule(0.25, 0.75, 19, 20) == 0.75
    assert linear_schedule(0.25, 0.75, 99, 20) == 0.75


def test_curriculum_opponent_cadence_has_intermediate_values():
    cadences = [round(linear_schedule(20, 10, i, 20)) for i in range(20)]
    assert cadences[0] == 20
    assert cadences[-1] == 10
    assert 19 in cadences and 18 in cadences


def test_simulator_wakes_on_non_multiple_opponent_cadence():
    assert next_wakeup_tick(10, 6000, opponent_decide=19) == 19
    assert next_wakeup_tick(19, 6000, opponent_decide=19) == 20
    assert next_wakeup_tick(20, 6000, opponent_decide=19) == 30


def test_reward_keeps_terminal_result_dominant():
    worst_win = shaped_return("win", [0, 3], [0.0, 3.0])[0]
    best_draw = shaped_return("draw", [3, 0], [3.0, 0.0])[0]
    worst_draw = shaped_return("draw", [0, 3], [0.0, 3.0])[0]
    best_loss = shaped_return("loss", [3, 0], [3.0, 0.0])[0]
    assert worst_win > best_draw
    assert worst_draw > best_loss


def test_reward_distinguishes_close_and_heavy_losses():
    close = shaped_return("loss", [0, 1], [1.9, 2.0])[0]
    heavy = shaped_return("loss", [0, 3], [0.0, 3.0])[0]
    assert 0.0 <= heavy < close < 0.5


def test_reward_rejects_weights_that_can_make_a_loss_beat_a_draw():
    with pytest.raises(ValueError, match="terminal weight"):
        shaped_return("draw", [0, 0], [3.0, 3.0], 0.5, 0.2, 0.3)


def test_evaluation_score_prefers_results_then_crown_margin():
    one_win_one_loss = {"wins": 1, "draws": 0, "n": 2, "crown_diff": -1.0}
    two_draws = {"wins": 0, "draws": 2, "n": 2, "crown_diff": 0.0}
    assert evaluation_score(two_draws) > evaluation_score(one_win_one_loss)


def test_gate_distribution_masks_actions_and_sums_to_one():
    logits = np.asarray([1.0, 2.0, 3.0, -4.0, 0.5])
    mask = np.asarray([True, True, False, True, True])
    probs = gate_probs(logits, mask)
    assert probs[2] == 0.0
    assert np.isclose(probs.sum(), 1.0)
    assert (probs[mask] > 0).all()


def test_unsupported_card_scan_checks_both_decks():
    battle = {
        "team": {"players": [{"deck": [{"card_key": "tesla"}]}]},
        "opponent": {"players": [{"deck": [{"card_key": "void"}]}]},
    }
    assert unsupported_deck_cards(battle, {"tesla": "Tesla"}) == {"void"}
