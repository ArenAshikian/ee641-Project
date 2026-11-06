# EE 641 Chess Bot - Live Lichess (Latest S49)

This runs the **existing, frozen Latest S49** against Lichess challengers. It uses
`ppo_train.one_move(..., sample=False)`, the same one-network-forward-pass greedy
move selection as the Maia evaluation, and `python-chess` for legal moves. It
**does not train, search, call Stockfish/Lc0, use opening books, or query
endgame tablebases**.

## Install

Extract the ZIP over `B:\EE641_Chess_RL_Starter\`, replacing `run.cmd` and adding
`chess_rl\lichess_live.py` and `tests\test_lichess_live.py`.

The saved checkpoint is used **read-only**:

`checkpoints\ppo_scale_latest\s0049.pt`

No model files or previous experiment outputs are changed.

## 1. Verify the neural policy locally

```cmd
cd /d B:\EE641_Chess_RL_Starter
run.cmd lichess-selftest
```

This makes 4 legal neural-network moves offline and shows the checkpoint SHA-256.
If your installed project includes tests, `run.cmd tests` also runs the new
Lichess adapter checks.

## 2. Create a separate Lichess bot account

**Do not convert your personal chess account.** Create a **new** Lichess account
that has never played a game and validate its email address. BOT conversion is
irreversible and the account can no longer play regular human-account games.

Under that separate account, visit https://lichess.org/account/oauth/token and
create a personal access token with the **bot:play** scope ("Play games with the
Bot API"). Keep it secret. Never paste the token into ChatGPT or a shared log.

In Command Prompt (not PowerShell):

```cmd
cd /d B:\EE641_Chess_RL_Starter
set /p LICHESS_BOT_TOKEN=Paste the new bot account token here: 
```

The token is kept in this Command Prompt session. It is **not** stored in the
project. `set /p` echoes what you type in the terminal, so don't share screenshots
of that line.

Check the account's status:

```cmd
run.cmd lichess-check
```

If it reports a non-BOT account, and you have verified it is the **new, zero-game
account**, run:

```cmd
run.cmd lichess-upgrade
```

The command warns you and requires typing `UPGRADE` before making the irreversible
API request. After successful conversion, run `run.cmd lichess-check` again.

## 3. First real live test (casual 3+2)

Optional, strongly recommended for the first game: only allow challenges from
your own regular Lichess username. Replace `YOUR_NORMAL_USERNAME`:

```cmd
set LICHESS_ALLOWED_CHALLENGER=YOUR_NORMAL_USERNAME
run.cmd lichess-play
```

The program prints a link like `https://lichess.org/@/YOUR_BOT_ACCOUNT`.

From your **regular personal account**, send the bot a **Standard**, **Casual**,
**3+2 Blitz** challenge. The bot accepts and plays using Latest S49.

If you want to accept challenges from anyone, run in the same Command Prompt:

```cmd
set LICHESS_ALLOWED_CHALLENGER=
run.cmd lichess-play
```

Press Ctrl+C to stop. Keep the computer and Command Prompt awake while playing.

## 4. Rated testing later

When the casual smoke game works, enable 3+2 **Rated** challenges in the same
Command Prompt before launching the bot:

```cmd
set LICHESS_ALLOW_RATED=1
run.cmd lichess-play
```

Rated games can change the bot's public Lichess Blitz rating. Use casual testing
first, and follow Lichess fair-play and bot-API rules. Bots only play challenge
games, not the normal automatic matchmaking pools.

## Research records

Each completed game is stored once in:

`outputs\lichess_latest_s49_games.jsonl`

The log contains the opponent, clock, rating status, full UCI moves, result,
model name, and SHA-256 of the exact checkpoint. This permits later offline
analysis. Live games do not update network weights.

## Common issues

- `Missing frozen Latest S49 checkpoint`: make sure the original PPO-scale
  experiment finished and `checkpoints\ppo_scale_latest\s0049.pt` exists.
- `title=(none)`: the new account is not yet upgraded to BOT.
- `HTTP 401/403`: your token is wrong, revoked, lacks bot:play, or the account
  has not been converted.
- Challenge declined as `timeControl`: select Standard 3+2, not a different
  clock or game type.
- Challenge declined as `casual`: rated is disabled until you opt in.
- Challenge declined as `generic`: set `LICHESS_ALLOWED_CHALLENGER` to the
  exact challenger username or clear it.
- If a stream disconnects, it reconnects automatically. If you disconnect the
  program entirely, you can relaunch it to resume any active game.

## Lichess references

- https://lichess.org/api (Bot endpoints and requirements)
- https://lichess.org/page/api-tips (rate-limit rules)
- https://lichess.org/account/oauth/token (create a minimal-scope API token)
