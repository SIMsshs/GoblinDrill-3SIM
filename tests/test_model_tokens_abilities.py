import torch

from sim.model_tokens import TokenPolicy, losses


def batch(n=3):
    b = {
        "unit_ids": torch.zeros(n, 48, dtype=torch.long),
        "unit_feats": torch.zeros(n, 48, 7),
        "towers": torch.ones(n, 6), "scalars": torch.zeros(n, 4),
        "hand": torch.tensor([[2, 3, 4, 5]] * n),
        "next": torch.full((n,), 6, dtype=torch.long),
        "opp_seen": torch.zeros(n, 8, dtype=torch.long),
        "opp_last": torch.zeros(n, 4, dtype=torch.long),
        "opp_last_age": torch.ones(n, 4),
        "ability_ids": torch.tensor([[7, 0, 0]] * n),
        "ability_feats": torch.zeros(n, 3, 4),
        "act_card": torch.tensor([0, 3, 0]),
        "act_x": torch.tensor([0, 4, 0]), "act_y": torch.tensor([0, 12, 0]),
        "act_gate": torch.tensor([0, 2, 5]),
        "outcome": torch.zeros(n), "won": torch.tensor([0.0, 1.0, 1.0]),
    }
    b["ability_feats"][:, 0, 0] = 1
    return b


def test_ability_gate_forward_and_backward():
    model = TokenPolicy(32, 32, d=24, layers=1, heads=4,
                        ability_actions=3, ability_features=4)
    b = batch()
    gate, tile, value = model(b)
    assert gate.shape == (3, 8)
    assert tile.shape == (3, 18 * 32)
    assert value.shape == (3, 2)
    loss, stats = losses(model, b, ability_weight=4.0)
    assert torch.isfinite(loss)
    assert {"gate_acc", "card_acc", "ability_acc", "tile_near"} <= stats.keys()
    loss.backward()
    assert model.ability_query.weight.grad is not None


def test_legacy_model_still_has_five_gate_actions():
    model = TokenPolicy(32, 32, d=24, layers=1, heads=4)
    b = batch()
    b.pop("ability_ids")
    b.pop("ability_feats")
    gate, _, _ = model(b)
    assert gate.shape == (3, 5)
