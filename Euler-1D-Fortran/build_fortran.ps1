Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Move to the folder containing this script.
Set-Location -Path $PSScriptRoot

Write-Host "[1/3] Cleaning previous build artifacts..."
if (-not (Test-Path -Path "build" -PathType Container)) {
    New-Item -Path "build" -ItemType Directory | Out-Null
}

Remove-Item -Path "build\*.o" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "build\*.mod" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "rocket_tester.exe" -Force -ErrorAction SilentlyContinue

Write-Host "[2/3] Compiling without OpenMP..."
& gfortran -O3 -march=native -ffast-math -J build -c input_vegae.f90 -o build\input_vegae.o
if ($LASTEXITCODE -ne 0) { throw "Build failed while compiling input_vegae.f90." }

& gfortran -O3 -march=native -ffast-math -J build -c riemann_test_cases.f90 -o build\riemann_test_cases.o
if ($LASTEXITCODE -ne 0) { throw "Build failed while compiling riemann_test_cases.f90." }

& gfortran -O3 -march=native -ffast-math -J build -c Rocket_tester.f90 -o build\Rocket_tester.o
if ($LASTEXITCODE -ne 0) { throw "Build failed while compiling Rocket_tester.f90." }

& gfortran build\input_vegae.o build\riemann_test_cases.o build\Rocket_tester.o -o rocket_tester.exe
if ($LASTEXITCODE -ne 0) { throw "Build failed while linking rocket_tester.exe." }

Write-Host "[3/3] Build completed: rocket_tester.exe"
