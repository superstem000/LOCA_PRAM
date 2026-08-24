# Copies the LAST-epoch checkpoint from every runs/tanh_*/ folder into
# runs/sweep/tanh_<value>.pth, ready for synthetic_eval.py.
#
# "Last" = the highest LOCA_PRAM_checkpoint_epoch_XXXX.pth in that folder.
# Copies (not moves) so the source dirs stay intact. Overwrite is enabled
# so re-running picks up new final-epoch checkpoints.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$RunsDir  = "runs"
$SweepDir = Join-Path $RunsDir "sweep"
New-Item -ItemType Directory -Force -Path $SweepDir | Out-Null

$foundAny = $false
foreach ($runDir in Get-ChildItem -Path $RunsDir -Directory -Filter "tanh_*") {
    if ($runDir.Name -eq "sweep") { continue }

    # Parse tanh value: everything between "tanh_" and the next "_"
    $match = [regex]::Match($runDir.Name, '^tanh_(\d+\.\d+)')
    if (-not $match.Success) {
        Write-Host "  [skip] can't parse tanh_scale from $($runDir.Name)"
        continue
    }
    $tsVal = $match.Groups[1].Value

    # Find highest-epoch checkpoint
    $ckpts = Get-ChildItem -Path $runDir.FullName -Filter "LOCA_PRAM_checkpoint_epoch_*.pth" -ErrorAction SilentlyContinue
    if ($ckpts.Count -eq 0) {
        Write-Host "  [skip] no LOCA_PRAM_checkpoint_epoch_*.pth in $($runDir.Name)"
        continue
    }
    # Sort by embedded epoch number (numeric, not lexical)
    $highest = $ckpts | ForEach-Object {
        $m = [regex]::Match($_.Name, 'epoch_(\d+)\.pth')
        [PSCustomObject]@{ File = $_; Epoch = if ($m.Success) { [int]$m.Groups[1].Value } else { 0 } }
    } | Sort-Object Epoch -Descending | Select-Object -First 1

    $dest = Join-Path $SweepDir ("tanh_{0}.pth" -f $tsVal)
    Write-Host ("  {0}" -f $runDir.Name)
    Write-Host ("    highest ckpt : {0}" -f $highest.File.Name)
    Write-Host ("    -> {0}" -f $dest)
    Copy-Item -Force -Path $highest.File.FullName -Destination $dest
    $foundAny = $true
}

if (-not $foundAny) {
    Write-Error "no runs/tanh_*/ folders with checkpoints found."
    exit 1
}

Write-Host ""
Write-Host ("Done. Files in {0}/:" -f $SweepDir)
Get-ChildItem -Path $SweepDir | Format-Table Name, Length, LastWriteTime
Write-Host ""
Write-Host "Next: python synthetic_eval.py"
