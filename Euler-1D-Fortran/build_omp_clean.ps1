Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Move to the folder containing this script.
Set-Location -Path $PSScriptRoot

Write-Host "Setting OpenMP environment variable: OMP_NUM_THREADS=16"
$env:OMP_NUM_THREADS = 16

Write-Host "[1/4] Cleaning previous build artifacts..."
if (-not (Test-Path -Path "build" -PathType Container)) {
    New-Item -Path "build" -ItemType Directory | Out-Null
}

Remove-Item -Path "build\*.o" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "build\*.mod" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "rocket_tester_omp.exe" -Force -ErrorAction SilentlyContinue

Write-Host "[2/4] Cleaning previous simulation output files..."
Remove-Item -Path "rocket_geometry_history_case*.dat" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "rocket_state_history_case*.dat" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "rocket_profiles_case*.dat" -Force -ErrorAction SilentlyContinue

Write-Host "[3/4] Compiling with OpenMP..."
& gfortran -O3 -march=native -ffast-math -fopenmp -J build -c input_vegae.f90 -o build\input_vegae_omp.o
if ($LASTEXITCODE -ne 0) { throw "Build failed while compiling input_vegae.f90." }

& gfortran -O3 -march=native -ffast-math -fopenmp -J build -c riemann_test_cases.f90 -o build\riemann_test_cases_omp.o
if ($LASTEXITCODE -ne 0) { throw "Build failed while compiling riemann_test_cases.f90." }

& gfortran -O3 -march=native -ffast-math -fopenmp -J build -c Rocket_tester_parallel.f90 -o build\Rocket_tester_omp.o
if ($LASTEXITCODE -ne 0) { throw "Build failed while compiling Rocket_tester_parallel.f90." }

& gfortran -fopenmp build\input_vegae_omp.o build\riemann_test_cases_omp.o build\Rocket_tester_omp.o -o rocket_tester_omp.exe
if ($LASTEXITCODE -ne 0) { throw "Build failed while linking rocket_tester_omp.exe." }

# & gfortran -O3 -march=native -ffast-math -fopenmp -J build -c teste.f90 -o build\teste_omp.o
# if ($LASTEXITCODE -ne 0) { throw "Build failed while compiling teste.f90." } 

# & gfortran -fopenmp build\teste_omp.o -o teste.exe
# if ($LASTEXITCODE -ne 0) { throw "Build failed while linking teste.exe." }



Write-Host "[4/4] Build completed: rocket_tester_omp.exe"
