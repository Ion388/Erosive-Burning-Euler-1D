Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Move to the folder containing this script.
Set-Location -Path $PSScriptRoot

if (-not $env:OMP_NUM_THREADS) {
    $env:OMP_NUM_THREADS = $env:4 # Best results with 4 threads for nx=200
}
if (-not $env:OMP_STACKSIZE) {
    $env:OMP_STACKSIZE = '256M'
}
Write-Host "OpenMP runtime: OMP_NUM_THREADS=$($env:OMP_NUM_THREADS), OMP_STACKSIZE=$($env:OMP_STACKSIZE)"

Write-Host "[1/4] Cleaning previous build artifacts..."
if (-not (Test-Path -Path "build" -PathType Container)) {
    New-Item -Path "build" -ItemType Directory | Out-Null
}

Remove-Item -Path "build\*.o" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "build\*.mod" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "rocket_solver.exe" -Force -ErrorAction SilentlyContinue

Write-Host "[2/4] Cleaning previous simulation output files..."
Remove-Item -Path "rocket_geometry_history_case*.dat" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "rocket_state_history_case*.dat" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "rocket_profiles_case*.dat" -Force -ErrorAction SilentlyContinue

Write-Host "[3/4] Compiling with OpenMP..."
& gfortran -O3 -march=native -ffast-math -fopenmp -J build -c input_vegae.f90 -o build\input_vegae_omp.o
if ($LASTEXITCODE -ne 0) { throw "Build failed while compiling input_vegae.f90." }

& gfortran -O3 -march=native -ffast-math -fopenmp -J build -c riemann_test_cases.f90 -o build\riemann_test_cases_omp.o
if ($LASTEXITCODE -ne 0) { throw "Build failed while compiling riemann_test_cases.f90." }

& gfortran -O3 -march=native -ffast-math -fopenmp -Wno-tabs -J build -c Rocket_tester_parallel.f90 -o build\Rocket_tester_parallel_omp.o
if ($LASTEXITCODE -ne 0) { throw "Build failed while compiling Rocket_tester_parallel.f90." }

& gfortran -fopenmp build\input_vegae_omp.o build\riemann_test_cases_omp.o build\Rocket_tester_parallel_omp.o -o rocket_solver.exe
if ($LASTEXITCODE -ne 0) { throw "Build failed while linking rocket_solver.exe." }

Write-Host "[4/4] Build completed: rocket_solver.exe"

.\rocket_solver.exe