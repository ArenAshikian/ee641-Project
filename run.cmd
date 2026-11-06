@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 goto failed
)

REM Keep existing CUDA-enabled PyTorch. This stamp is created by the earlier update.
if not exist ".venv\.requirements_phase1_installed" (
    echo Installing project requirements...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 goto failed
    type nul > ".venv\.requirements_phase1_installed"
)

if /I "%~1"=="ppo-historical-smoke" (
    echo Checking original-V100 historical-branch tests...
    .venv\Scripts\python.exe -m unittest discover -s tests -p test_ppo_historical.py -q
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue smoke
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-historical-37k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue train --steps 49
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-historical-50k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue train --steps 75
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-historical-75k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue train --steps 124
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-historical-100k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue train --steps 173
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-historical-200k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue train --steps 367
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-historical-300k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue train --steps 563
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-historical-400k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue train --steps 758
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-historical-500k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue train --steps 953
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-historical-600k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue train --steps 1149
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-historical-1m" (
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue train --steps 1930
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-historical-report" (
    .venv\Scripts\python.exe -m chess_rl.ppo_historical_continue report
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-smoke" (
    echo Checking isolated batch-continuation tests...
    .venv\Scripts\python.exe -m unittest discover -s tests -p test_ppo_batch_continue.py -q
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue smoke
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-50k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue train --steps 26
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-75k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue train --steps 75
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-100k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue train --steps 124
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-150k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue train --steps 221
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-200k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue train --steps 318
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-300k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue train --steps 514
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-400k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue train --steps 709
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-500k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue train --steps 904
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-600k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue train --steps 1100
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-750k" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue train --steps 1393
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-1m" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue train --steps 1881
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-continue-report" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_continue report
    if errorlevel 1 goto failed
) else if /I "%~1"=="research-snapshot" (
    .venv\Scripts\python.exe -m chess_rl.research_snapshot
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-batch-smoke" (
    echo Checking batched PPO correctness tests...
    .venv\Scripts\python.exe -m unittest discover -s tests -p test_ppo_batch.py -q
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_bench --smoke
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-batch-bench" (
    .venv\Scripts\python.exe -m chess_rl.ppo_batch_bench --games 32 --max-plies 80 --batch-size 32
    if errorlevel 1 goto failed
) else if /I "%~1"=="maia-confirm-prepare" (
    if "%~2"=="" (
        .venv\Scripts\python.exe -m chess_rl.maia_confirm prepare
    ) else (
        .venv\Scripts\python.exe -m chess_rl.maia_confirm prepare --source "%~2"
    )
    if errorlevel 1 goto failed
) else if /I "%~1"=="maia-confirm-smoke" (
    echo Checking fresh Maia confirmation tests...
    .venv\Scripts\python.exe -m unittest discover -s tests -p test_maia_confirm.py -q
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.maia_confirm smoke
    if errorlevel 1 goto failed
) else if /I "%~1"=="maia-confirm-eval" (
    .venv\Scripts\python.exe -m chess_rl.maia_confirm eval
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-milestone-smoke" (
    .venv\Scripts\python.exe -m unittest discover -s tests -p test_ppo_milestone_eval.py -q
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.ppo_milestone_eval smoke
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-milestone-eval" (
    .venv\Scripts\python.exe -m chess_rl.ppo_milestone_eval eval
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-milestone-report" (
    .venv\Scripts\python.exe -m chess_rl.ppo_milestone_eval report
    if errorlevel 1 goto failed
) else if /I "%~1"=="maia-confirm-report" (
    .venv\Scripts\python.exe -m chess_rl.maia_confirm report
    if errorlevel 1 goto failed
) else if /I "%~1"=="lichess-selftest" (
    .venv\Scripts\python.exe -m chess_rl.lichess_live selftest
    if errorlevel 1 goto failed
) else if /I "%~1"=="lichess-check" (
    .venv\Scripts\python.exe -m chess_rl.lichess_live check
    if errorlevel 1 goto failed
) else if /I "%~1"=="lichess-upgrade" (
    .venv\Scripts\python.exe -m chess_rl.lichess_live upgrade
    if errorlevel 1 goto failed
) else if /I "%~1"=="lichess-play" (
    .venv\Scripts\python.exe -m chess_rl.lichess_live play
    if errorlevel 1 goto failed
) else if /I "%~1"=="maia-download" (
    .venv\Scripts\python.exe -m chess_rl.maia_benchmark --mode download
    if errorlevel 1 goto failed
) else if /I "%~1"=="maia-prepare" (
    .venv\Scripts\python.exe -m chess_rl.maia_benchmark --mode prepare %2 %3 %4
    if errorlevel 1 goto failed
) else if /I "%~1"=="maia-smoke" (
    echo Checking tests before one-node Maia chess integration...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.maia_benchmark --mode smoke
    if errorlevel 1 goto failed
) else if /I "%~1"=="maia-eval" (
    .venv\Scripts\python.exe -m chess_rl.maia_benchmark --mode eval
    if errorlevel 1 goto failed
) else if /I "%~1"=="maia-eval-full" (
    .venv\Scripts\python.exe -m chess_rl.maia_benchmark --mode full
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-scale" (
    echo Checking tests before new search-free PPO scaling study...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    echo Running TWO 25,088-game PPO continuations from frozen V100...
    echo Pool branch first, then latest-only; each branch resumes independently.
    .venv\Scripts\python.exe -m chess_rl.ppo_scale --mode both
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-scale-smoke" (
    echo Checking tests before integrated PPO smoke test...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.ppo_scale --mode smoke
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-scale-prepare" (
    .venv\Scripts\python.exe -m chess_rl.ppo_scale --mode prepare
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-scale-pool" (
    echo Checking tests before historical-pool PPO training...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.ppo_scale --mode pool
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-scale-latest" (
    echo Checking tests before latest-only PPO training...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.ppo_scale --mode latest
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-scale-eval" (
    .venv\Scripts\python.exe -m chess_rl.ppo_scale --mode eval
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-scale-report" (
    .venv\Scripts\python.exe -m chess_rl.ppo_scale --mode report
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-temperature" (
    echo Checking tests before no-search temperature experiment...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    echo Comparing V0/V100 sampling temperatures and matched entropy, NO SEARCH...
    .venv\Scripts\python.exe -m chess_rl.ppo_temperature_test
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-policy-test" (
    echo Checking tests before matched PPO policy comparison...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    echo Evaluating V0 and V100 under greedy and sampled play, NO SEARCH...
    .venv\Scripts\python.exe -m chess_rl.ppo_policy_test
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-diagnose" (
    echo Checking tests before search-free PPO diagnostics...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    echo Running read-only plateau diagnostics, no search or engine...
    .venv\Scripts\python.exe -m chess_rl.ppo_diagnostics
    if errorlevel 1 goto failed
) else if /I "%~1"=="championship" (
    echo Checking tests before search-free checkpoint championship...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    echo Starting five paired-color matches against frozen neural checkpoints...
    .venv\Scripts\python.exe -m chess_rl.checkpoint_championship
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-long" (
    echo Checking tests before extended search-free PPO training...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    echo Continuing PPO from permanent V10; 128 games per iteration to V100...
    .venv\Scripts\python.exe -m chess_rl.ppo_long
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-long-eval" (
    .venv\Scripts\python.exe -m chess_rl.ppo_long --eval-only
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-pilot" (
    echo Checking tests before search-free PPO pilot...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    echo Starting PPO correctness pilot with no search or engine...
    .venv\Scripts\python.exe -m chess_rl.ppo_train --mode pilot
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-train" (
    echo Checking tests before search-free PPO training...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    echo Starting PPO self-play from permanent V0...
    .venv\Scripts\python.exe -m chess_rl.ppo_train --mode train
    if errorlevel 1 goto failed
) else if /I "%~1"=="ppo-eval" (
    .venv\Scripts\python.exe -m chess_rl.ppo_train --mode eval
    if errorlevel 1 goto failed
) else if /I "%~1"=="train-v0" (
    echo Checking tests before full supervised V0 training...
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    echo Starting permanent 20-million-position supervised training...
    .venv\Scripts\python.exe -m chess_rl.train_v0
    if errorlevel 1 goto failed
) else if /I "%~1"=="prep-v0" (
    .venv\Scripts\python.exe -m unittest discover -s tests -q
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.benchmark_v0
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.evaluate_v0
    if errorlevel 1 goto failed
) else if /I "%~1"=="benchmark-v0" (
    .venv\Scripts\python.exe -m chess_rl.benchmark_v0
    if errorlevel 1 goto failed
) else if /I "%~1"=="eval-v0" (
    .venv\Scripts\python.exe -m chess_rl.evaluate_v0 %2 %3 %4 %5
    if errorlevel 1 goto failed
) else if /I "%~1"=="train-1m" (
    .venv\Scripts\python.exe -m chess_rl.train_1m
    if errorlevel 1 goto failed
) else if /I "%~1"=="train-100k" (
    .venv\Scripts\python.exe -m chess_rl.train_100k
    if errorlevel 1 goto failed
) else if /I "%~1"=="train" (
    .venv\Scripts\python.exe -m chess_rl.train_supervised
    if errorlevel 1 goto failed
) else if /I "%~1"=="tests" (
    .venv\Scripts\python.exe -m unittest discover -s tests -v
    if errorlevel 1 goto failed
) else if "%~1"=="" (
    .venv\Scripts\python.exe -m unittest discover -s tests -v
    if errorlevel 1 goto failed
    .venv\Scripts\python.exe -m chess_rl.main model
    if errorlevel 1 goto failed
    echo Next recommended command: run.cmd train-v0
) else (
    .venv\Scripts\python.exe -m chess_rl.main %*
    if errorlevel 1 goto failed
)

echo.
echo Finished successfully.
exit /b 0

:failed
echo.
echo Something failed. See the error above.
exit /b 1
