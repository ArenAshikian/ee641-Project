EE 641 Chess RL: PPO no-learner-turn bug fix

What happened:
A game ended before the learner made a move. The PPO result for that game
was missing outcome_for_learner, and the result counter crashed.

Install:
1. Copy chess_rl/ppo_train.py and tests/test_ppo_terminal_before_learner.py
   from this ZIP into B:\EE641_Chess_RL_Starter\, keeping the folder paths.
2. Replace the old chess_rl/ppo_train.py when Windows asks.
3. Open CMD in B:\EE641_Chess_RL_Starter\ and run:

   run.cmd ppo-train

The existing launcher first runs all tests, including the new regression tests.
It then loads checkpoints/ppo_train_latest.pt and resumes at iteration 5
if iterations 1-4 were successfully saved, as indicated by your log.

Do not delete any checkpoints or the training CSV, do not rerun train-v0,
and do not change the PPO seed or configuration.

The fix does not change the neural network, PPO math, self-play sampling,
legal-move masking, or the no-search design. Terminal results with no learner
moves are counted, but provide no learner transitions for optimization.
