param(
    [string]$Arch = "sm_89"
)

$ErrorActionPreference = "Stop"

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    throw "nvcc introuvable dans le PATH. Installez le CUDA Toolkit Windows (ex. 12.4+) et lancez depuis un shell avec MSVC/CUDA."
}

$root = Split-Path -Parent $PSScriptRoot
$appDir = Join-Path $root "app"
$cuFile = Join-Path $appDir "crates\aibox-kernels\cuda\aibox_ternary_gpu.cu"
$outDir = Join-Path $appDir "target\release"
$dllOut = Join-Path $outDir "aibox_gpu.dll"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "Compilation de aibox_gpu.dll ($Arch)..."
& nvcc -O3 "-arch=$Arch" --shared $cuFile -o $dllOut
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Compilation release Rust (aibox.exe)..."
& cargo build --release --manifest-path (Join-Path $appDir "Cargo.toml")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Build termine : $dllOut & $(Join-Path $outDir 'aibox.exe')"
