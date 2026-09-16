# Hand the GPU from the running replication straight to the block queue.
#
# The replication (scripts/15_run_parallel.py --grid replication) still has hours to
# go, and scripts/18_queue.py must not start until it has released the GPU: measured
# throughput is 0.0357 epochs/s with one worker against 0.0284 with two, so two
# training processes on this card make both slower, not faster.
#
# Rather than waiting for someone to notice the replication has finished, this
# watches for the training process and starts the queue the moment it exits.  That
# makes the remaining ~20 h unattended.
#
# Usage:
#     pwsh -File scripts\run_queue_after_training.ps1
#     pwsh -File scripts\run_queue_after_training.ps1 -MaxWaitHours 14 -Blocks fxu signed
#
# The queue itself is resumable (it skips blocks with a success marker), so running
# this twice is harmless -- but two concurrent queues would share the GPU badly, so
# it takes a lock first.

param(
    [string]$Python = "E:\anaconda3\envs\pytorch-clean\python.exe",
    [string]$WatchFor = "15_run_parallel",
    [double]$MaxWaitHours = 14,
    [string[]]$Blocks = @(),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

$lock = Join-Path $root "data\processed\queue\chain.lock"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $lock) | Out-Null

function Write-Stamp($message) {
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $message)
}

# ---- lock ------------------------------------------------------------------
if (Test-Path $lock) {
    $holder = (Get-Content $lock -Raw).Trim()
    $alive = Get-Process -Id $holder -ErrorAction SilentlyContinue
    if ($alive) {
        Write-Stamp "another chain is already armed (pid $holder); exiting"
        exit 0
    }
    Write-Stamp "clearing a stale lock from pid $holder"
}
$PID | Out-File -FilePath $lock -Encoding ascii
try {

    # ---- wait for the GPU ---------------------------------------------------
    # The driver is `python scripts\15_run_parallel.py ...` but the process that
    # actually holds the GPU is its multiprocessing spawn child, whose command line
    # does NOT contain the script name.  Matching only the driver would start the
    # queue alongside a lingering worker, and two training processes on this card are
    # slower than one (0.0284 against 0.0357 epochs/s).  Match both.
    #
    # If the deadline expires while training is STILL running, do not start the queue:
    # that is the exact situation this guard exists to prevent, and starting anyway
    # would silently make both jobs slower and produce timing metrics from a contended
    # GPU.  Abort instead and say so.
    $deadline = (Get-Date).AddHours($MaxWaitHours)
    $waited = 0
    $running = $null
    while ($true) {
        $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                     Where-Object {
                         $_.CommandLine -and (
                             $_.CommandLine -like "*$WatchFor*" -or
                             $_.CommandLine -like "*--multiprocessing-fork*")
                     })
        if ($running.Count -eq 0) { break }
        if ($waited -eq 0) {
            Write-Stamp ("waiting for {0} pid(s) {1} to release the GPU" -f
                         $running.Count, (($running | ForEach-Object { $_.ProcessId }) -join ","))
        }
        if ((Get-Date) -ge $deadline) {
            Write-Stamp ("ABORT: {0} still running after {1} min; not starting the queue" -f
                         $running.Count, $waited)
            exit 3
        }
        Start-Sleep -Seconds 60
        $waited += 1
        if ($waited % 15 -eq 0) { Write-Stamp ("still waiting ({0} min)" -f $waited) }
    }
    if ($waited -gt 0) { Write-Stamp ("GPU released after {0} min" -f $waited) }

    # ---- run the queue -----------------------------------------------------
    $args = @("scripts\18_queue.py")
    if ($Blocks.Count -gt 0) { $args += @("--blocks") + $Blocks }
    if ($Force) { $args += "--force" }
    Write-Stamp ("starting the block queue: {0} {1}" -f $Python, ($args -join " "))
    & $Python @args
    $rc = $LASTEXITCODE
    Write-Stamp ("queue exited with code $rc")
    exit $rc
}
finally {
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
}
