# Goblin Drill training checkpoint report — 2026-10-13

## Decision

Keep `runs/tokens/drill_target_bc/best.pt` as the active policy.  It is also
copied unchanged to `runs/selfplay/drill_selfplay_safe_v2/best.pt` as the
protected `it000.pt` baseline.  Do not deploy `it040.pt` from the first
self-play run or `it002.pt` from the safe trial.

## Fair shuffled-start evaluation

All policies below played the same 52 trials (13 held-out opponent-deck
templates, simulator seeds 9000–9051, both starting decks shuffled, 21-tick
command delay, -21-tick rebuilt-observation lead) against
`runs/tokens/drill_base_bc/best.pt`.

| policy | wins | draws | result score | crown diff | tower diff | plays/match |
|---|---:|---:|---:|---:|---:|---:|
| base BC | 8 | 3 | 18.3% | -1.000 | -0.748 | 23.7 |
| target BC | 13 | 1 | **26.0%** | -0.846 | -0.674 | 25.3 |
| first self-play best (`it005`) | 12 | 0 | 23.1% | **-0.673** | **-0.620** | 27.4 |
| first self-play final (`it040`) | 6 | 6 | 17.3% | -0.788 | -0.641 | 24.7 |

The earlier 13-match selector overrated `it005`.  The larger fixed-seed screen
puts target BC first on game result.  `it005` preserves towers slightly better,
but does not convert that into more wins.

## Conservative self-play trial

`drill_selfplay_safe_v2` used target BC for both `--init` and the KL reference,
`lr=2e-6`, `kl=0.15`, two iterations, 48 collection matches per iteration,
and a 35% to 45% reactive-opponent curriculum.

The protected baseline reproduced 13 wins and one draw (26.0%).  `it002.pt`
then scored only 7 wins and one draw (14.4%), with crown diff -1.077.  The
promotion guard correctly retained `it000.pt` as `best.pt`.

## Rolex CR visual data

The extracted `XEGIAnHNiCw` set contains seven matches, 423 card actions, three
confirmed Hero Knight ability actions, and 541 wait samples.  A small visual BC
baseline trained for 30 epochs.  Its held-out test accuracy was 58.0% for
wait/card/ability kind and 17.9% for card identity (12.5% random-card baseline).
This is an ingestion proof, not a policy suitable for deployment or fusion.

The next useful milestone is more expert data rather than more PPO: at least 50
same-deck matches for the first pass, exact positions for Drill/Poison/Tesla/
Snowball, and enough Hero Knight ability examples to evaluate that action.

## Evaluator safeguards added

- `--matches` can now produce the requested number of reactive trials by
  cycling held-out deck templates with new simulator seeds.
- `--shuffle-starts` randomizes both hands and is rejected for recorded-ghost
  evaluation, where shuffled hands would invalidate recorded commands.
- trial seed, template count, losses, and tower difference are written to JSON.
- self-play evaluates and protects its starting checkpoint before any update;
  a worse first candidate can no longer become `best.pt` automatically.
