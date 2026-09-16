# End-to-end pipeline for the fly-connectome numerosity study.
#
# Usage (from the project root):
#     pwsh -File scripts\run_all.ps1              # full pipeline
#     pwsh -File scripts\run_all.ps1 -Stage 0     # only Stage 0
#     pwsh -File scripts\run_all.ps1 -Stage 2     # only the learning curve
#
# Every step is idempotent: downloaded files are skipped, circuits and stimuli
# are cached, and a configuration whose results already exist gets a fresh
# "_rN" run directory rather than appending to the old one.

param(
    [string]$Stage = "all",
    [string]$Python = "E:\anaconda3\envs\pytorch-clean\python.exe",
    [int]$MaxEpochs = 60,
    [int]$Patience = 15
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

function Step($name, $script, $args) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host "  $name" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Python $script @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  STEP FAILED (exit $LASTEXITCODE): $script" -ForegroundColor Red
        throw "step failed: $script"
    }
    Write-Host ("  done in {0:N1} min" -f $sw.Elapsed.TotalMinutes) -ForegroundColor Green
}

$run = { param($s) $Stage -eq "all" -or $Stage -eq $s }

# ---------------------------------------------------------------- Stage 0 ---
if (& $run "0") {
    Step "0a  environment self-check"     "scripts\00_env_check.py"        @()
    Step "0b  download MaleCNS v1.0"      "scripts\01_download.py"         @()
    Step "0c  build edge list"            "scripts\02_build_connectome.py" @()
    Step "0d  circuits + retinal map"     "scripts\03_build_circuits.py"   @()
    Step "0f  stimuli + control audit"    "scripts\04_build_stimuli.py"    @()
    Step "0g  calibration + propagation"  "scripts\05_calibrate.py"        @()
    Step "0h  unit tests"                 "-m"                             @("pytest", "tests", "-q")
    Step "0i  non-connectome baselines"   "scripts\10_baselines.py"        @("--n-train", "8000")
}

# ---------------------------------------------------------------- Stage 1 ---
if (& $run "1") {
    Step "1   M1 taught fly (N=20000)"    "scripts\06_run_count.py" @(
        "--model", "M1", "--graph", "real", "--n-train", "20000",
        "--max-epochs", "$MaxEpochs", "--patience", "$Patience")
    Step "1   M0 frozen fly (N=20000)"    "scripts\06_run_count.py" @(
        "--model", "M0", "--graph", "real", "--n-train", "20000",
        "--max-epochs", "$MaxEpochs", "--patience", "$Patience")
}

# ---------------------------------------------------------------- Stage 2 ---
if (& $run "2") {
    Step "2   real-vs-shuffled learning curve (30 runs)" "scripts\07_run_learning_curve.py" @(
        "--models", "M1", "--n-grid", "100", "500", "1000", "5000", "20000",
        "--seeds", "0", "1", "2", "--graphs", "real", "shuffled",
        "--max-epochs", "$MaxEpochs", "--patience", "$Patience")
    Step "2   frozen-fly probe curve (M0)" "scripts\07_run_learning_curve.py" @(
        "--models", "M0", "--n-grid", "500", "5000", "20000",
        "--seeds", "0", "1", "2", "--graphs", "real",
        "--max-epochs", "$MaxEpochs", "--patience", "$Patience")
}

# ---------------------------------------------------------------- Stage 3 ---
# Conditions B/C/D are evaluated inside every counting run, so Stage 3 needs no
# extra training: the numbers are already in each run's summary.json.

# ---------------------------------------------------------------- Stage 4 ---
if (& $run "4") {
    Step "4   addition, real connectome"  "scripts\08_run_addition.py" @(
        "--graph", "real", "--all-seeds",
        "--max-epochs", "$MaxEpochs", "--patience", "$Patience")
    Step "4   addition, shuffled control" "scripts\08_run_addition.py" @(
        "--graph", "shuffled", "--all-seeds",
        "--max-epochs", "$MaxEpochs", "--patience", "$Patience")
}

# ---------------------------------------------------------------- Stage 6 ---
# Circuit-completeness ladder.  `full` keeps 91.3% of outgoing synaptic weight
# against core's 26.4%, but its degree heterogeneity (max/mean row sum 129 vs 59)
# forces row normalisation and a much smaller weight scale, which is why it is a
# separate stage rather than part of the main grid.
if (& $run "6") {
    Step "6   calibrate the full circuit"  "scripts\05_calibrate.py" @(
        "--circuit", "full", "--batch-size", "16", "--criterion", "stability")
    Step "6   full circuit, real"          "scripts\06_run_count.py" @(
        "--model", "M1", "--circuit", "full", "--graph", "real",
        "--n-train", "5000", "--batch-size", "16", "--w-scale", "0.4",
        "--normalization", "row", "--max-epochs", "40", "--patience", "12")
    Step "6   full circuit, shuffled"      "scripts\06_run_count.py" @(
        "--model", "M1", "--circuit", "full", "--graph", "shuffled",
        "--n-train", "5000", "--batch-size", "16", "--w-scale", "0.4",
        "--normalization", "row", "--max-epochs", "40", "--patience", "12")
}

# ---------------------------------------------------------------- Stage 5 ---
if (& $run "5") {
    Step "5   temporal decoding (memory)" "scripts\13_temporal.py" @("--graph", "real")
}

# ------------------------------------------------------------- reporting ---
if (& $run "figs") {
    Step "figures + report"               "scripts\09_analysis.py" @()
    Step "status table"                   "scripts\status.py"      @()
}

Write-Host ""
Write-Host "PIPELINE COMPLETE" -ForegroundColor Green
Write-Host "  runs   : $root\runs\index.csv"
Write-Host "  report : $root\reports\report.md"
Write-Host "  figures: $root\figures\"
