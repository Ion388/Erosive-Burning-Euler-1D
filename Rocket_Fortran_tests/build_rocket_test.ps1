param(
    [switch]$NoRun,
    [switch]$DebugBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot 'build'
$InputSource = Join-Path $ProjectRoot 'input_vegae.f90'
$RiemannSource = Join-Path $ProjectRoot 'riemann_test_cases.f90'
$GeometrySource = Join-Path $ProjectRoot 'grain_levelset_preprocessor.f90'
$SolverSource = Join-Path $ProjectRoot 'Rocket_tester_parallel.f90'

$InputObject = Join-Path $BuildDir 'input_vegae.o'
$RiemannObject = Join-Path $BuildDir 'riemann_test_cases.o'
$GeometryObject = Join-Path $BuildDir 'grain_levelset_preprocessor.o'
$SolverObject = Join-Path $BuildDir 'Rocket_tester_parallel.o'
$GeometryExe = Join-Path $BuildDir 'grain_levelset_preprocessor.exe'
$SolverExe = Join-Path $BuildDir 'rocket_solver.exe'
$GeometryTable = Join-Path $BuildDir 'grain_geometry_table.dat'

$Compiler = (Get-Command gfortran -ErrorAction Stop).Source

foreach ($RequiredSource in @($InputSource, $RiemannSource, $GeometrySource, $SolverSource)) {
    if (-not (Test-Path -LiteralPath $RequiredSource -PathType Leaf)) {
        throw "Required source file is missing: $RequiredSource"
    }
}

if (-not $env:OMP_NUM_THREADS) {
    $env:OMP_NUM_THREADS = '4'
}
if (-not $env:OMP_STACKSIZE) {
    $env:OMP_STACKSIZE = '256M'
}
Write-Host "OpenMP runtime: OMP_NUM_THREADS=$($env:OMP_NUM_THREADS), OMP_STACKSIZE=$($env:OMP_STACKSIZE)"

Write-Host '[1/7] Preparing clean build directory...'
if (Test-Path -LiteralPath $BuildDir -PathType Container) {
    Get-ChildItem -LiteralPath $BuildDir -Force | Remove-Item -Recurse -Force
}
else {
    New-Item -Path $BuildDir -ItemType Directory | Out-Null
}

$CompileFlags = @(
    '-fopenmp',
    '-fbacktrace',
    '-ffree-line-length-none',
    '-Wno-tabs',
    '-J', $BuildDir,
    '-I', $BuildDir
)
if ($DebugBuild) {
    $CompileFlags = @('-O0', '-g', '-fcheck=all', '-Wall', '-Wextra') + $CompileFlags
}
else {
    # Do not use -ffast-math: the solver deliberately checks for NaN/Inf states.
    $CompileFlags = @('-O3', '-march=native') + $CompileFlags
}

Write-Host '[2/7] Compiling input module...'
& $Compiler @CompileFlags -c $InputSource -o $InputObject
if ($LASTEXITCODE -ne 0) { throw 'Build failed while compiling input_vegae.f90.' }

Write-Host '[3/7] Compiling and linking level-set preprocessor...'
& $Compiler @CompileFlags -c $GeometrySource -o $GeometryObject
if ($LASTEXITCODE -ne 0) { throw 'Build failed while compiling grain_levelset_preprocessor.f90.' }
& $Compiler -fopenmp $InputObject $GeometryObject -o $GeometryExe
if ($LASTEXITCODE -ne 0) { throw 'Build failed while linking grain_levelset_preprocessor.exe.' }

Write-Host '[4/7] Compiling Riemann test-case module...'
& $Compiler @CompileFlags -c $RiemannSource -o $RiemannObject
if ($LASTEXITCODE -ne 0) { throw 'Build failed while compiling riemann_test_cases.f90.' }

Write-Host '[5/7] Compiling and linking rocket solver...'
& $Compiler @CompileFlags -c $SolverSource -o $SolverObject
if ($LASTEXITCODE -ne 0) { throw 'Build failed while compiling Rocket_tester_parallel.f90.' }
& $Compiler -fopenmp $InputObject $RiemannObject $SolverObject -o $SolverExe
if ($LASTEXITCODE -ne 0) { throw 'Build failed while linking rocket_solver.exe.' }

Write-Host '[6/7] Build completed.'
Write-Host "  Geometry preprocessor: $GeometryExe"
Write-Host "  Rocket solver:         $SolverExe"

if ($NoRun) {
    Write-Host '[7/7] Run skipped (-NoRun).'
    exit 0
}

Write-Host '[7/7] Generating geometry table and running solver...'
Push-Location $BuildDir
try {
    & $GeometryExe $GeometryTable
    if ($LASTEXITCODE -ne 0) { throw 'Geometry preprocessor failed.' }

    & $SolverExe $GeometryTable
    if ($LASTEXITCODE -ne 0) { throw 'Rocket solver failed.' }
}
finally {
    Pop-Location
}

Write-Host "Run completed. Generated files are in: $BuildDir"
