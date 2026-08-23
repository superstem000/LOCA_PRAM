# Sweeps --tanh-scale across {1.0, 2.0, 4.0, 6.0} sequentially, with everything
# else at train.py defaults (2000 epochs, seed=2, batch_size=16, lr=3e-5,
# defect BG folder). Same seed and schedule across all four runs so the
# only varying axis is tanh_scale. 6.0 is included so we get a same-budget
# comparison against the current inference default.
#
# Usage (from PowerShell):  ./ablate_tanh.ps1
# The script exits on the first failed run.

$ErrorActionPreference = "Stop"

$TS = Get-Date -Format "yyyyMMdd_HHmmss"
Set-Location -Path $PSScriptRoot

foreach ($TSVal in @("1.0", "2.0", "4.0", "6.0")) {
    $Out = "runs/tanh_${TSVal}_${TS}/"
    Write-Host "=========================================================="
    Write-Host ("[{0}] START tanh_scale={1}  ->  {2}" -f (Get-Date -Format "HH:mm:ss"), $TSVal, $Out)
    Write-Host "=========================================================="

    python train.py `
        --tanh-scale $TSVal `
        --output-dir $Out `
        --seed 2 `
        --max-epoch 2000

    if ($LASTEXITCODE -ne 0) {
        Write-Error ("train.py exited with code {0} for tanh_scale={1}" -f $LASTEXITCODE, $TSVal)
        exit $LASTEXITCODE
    }

    Write-Host ("[{0}] DONE  tanh_scale={1}" -f (Get-Date -Format "HH:mm:ss"), $TSVal)
}

Write-Host ""
Write-Host "All four runs finished. Compare with:"
Write-Host "  tensorboard --logdir runs/"
