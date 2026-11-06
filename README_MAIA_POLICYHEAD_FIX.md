# Maia benchmark: policyhead hotfix

The first smoke test found a real configuration problem: the Lc0 process was
using `Search algorithm: classic`, which may report two nodes despite a
`go nodes 1` request. The benchmark then stopped at `RuntimeError: ... limit: 2`.

Lc0 v0.32.1 supports **policyhead**, a special one-network-evaluation mode
which does **not** use MCTS. The Lc0 developers document that it can be
selected by renaming the executable to contain `policyhead`. This hotfix
copies your own existing `lc0.exe` to `lc0-policyhead.exe` automatically in
the same folder, preserving the original and its required DLLs.

The benchmark now verifies that it launches the policyhead executable. It
still requests `go nodes 1`, but treats Lc0's optional informational node
counter as diagnostic; it does not equate this counter with actual search
when policyhead is enabled.

## Install

Extract `EE641_Maia_Policyhead_Fix.zip` into your existing project folder:

    B:\EE641_Chess_RL_Starter\

Replace only:

    chess_rl\maia_benchmark.py
    tests\test_maia_benchmark.py

No model checkpoints, weights, original Lc0 executable, openings, or PPO
results are changed.

## Run

    run.cmd maia-smoke

In the output verify the Lc0 banner says:

    Search algorithm: policyhead

The 16-game smoke evaluation is intentionally saved under new filenames
`outputs\maia_policy_p002_*` so it **does not mix** the 4 games already
saved with the original `classic` engine into the new benchmark.

If smoke succeeds, run:

    run.cmd maia-eval

The 1,536-game standard evaluation now writes `outputs\maia_policy_p064_*`.
It is resumable after each completed game. The older `maia_p002_*` files are
left untouched for audit purposes.

## References

- Lc0 documentation: https://lczero.org/dev/overview/
- Lc0 changelog: https://github.com/LeelaChessZero/lc0/blob/master/changelog.txt
- Maia official repository: https://github.com/CSSLab/maia-chess

The entire evaluation is read-only with respect to our neural checkpoints,
and our own chess model continues to use one forward pass with no search.
