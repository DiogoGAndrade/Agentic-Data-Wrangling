# run_phase_b.ps1
# ----------------------------------------------------------------------
# Phase B: Generalization test on 3 held-out datasets.
# Runs 2 representative LLMs (cheapest + largest) × 3 datasets × 4 ML models.
# Uses C4 (expanded action space) with per-model context specs.
#
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File .\run_phase_b.ps1
#
# Pre-conditions:
#   - Ollama is running:   ollama serve
#   - Models pulled:  ollama pull qwen2.5:3b   and   ollama pull qwen2.5:14b
#   - Python venv active:  .\.venv\Scripts\Activate.ps1
#   - c0_raw.csv generated for each Phase B dataset
#   - C4 context specs exist in evaluation/c4_contexts/
# ----------------------------------------------------------------------

$ErrorActionPreference = "Stop"

# Two representative LLMs: cheapest (3B) and largest (14B)
# Phase A showed all 8 produce identical results — these two bracket the range.
$models = @(
    @{ ollama = "qwen2.5:3b";  tag = "qwen2.5_3b"  },
    @{ ollama = "qwen2.5:14b"; tag = "qwen2.5_14b"  }
)

# Phase B datasets and their ML models
$datasets = @{
    "house_prices" = @("ridge", "rf", "knn", "gbm")   # regression
    "heart"        = @("logreg", "rf", "knn", "gbm")   # classification
    "bank"         = @("logreg", "rf", "knn", "gbm")   # classification
}

$ollamaUrl   = "http://localhost:11434"
$contextsDir = "evaluation\c4_contexts"

Write-Host "===== PHASE B: 2 LLMs x 3 datasets x 4 ML models (C4 expanded) =====" -ForegroundColor Cyan
Write-Host "Datasets: house_prices (regression), heart (classification), bank (classification)" -ForegroundColor Cyan

$totalSteps = 0
$successSteps = 0
$failedSteps = @()

foreach ($m in $models) {
    $ollamaTag = $m.ollama
    $shortTag  = $m.tag
    Write-Host ""
    Write-Host "######################################################################" -ForegroundColor Yellow
    Write-Host "# LLM: $ollamaTag ($shortTag)" -ForegroundColor Yellow
    Write-Host "######################################################################" -ForegroundColor Yellow

    foreach ($ds in $datasets.Keys) {
        $mlModels = $datasets[$ds]

        foreach ($ml in $mlModels) {
            $totalSteps++
            $planTag  = "${shortTag}_${ml}"
            $ctxFile  = "$contextsDir\${ds}_${ml}.json"

            Write-Host ""
            Write-Host "----- $ds | $ml | $shortTag -----" -ForegroundColor Green

            # Check context file exists
            if (-not (Test-Path $ctxFile)) {
                Write-Host "[SKIP] Missing context: $ctxFile" -ForegroundColor Red
                $failedSteps += "${ds}|${ml}|${shortTag}: missing context"
                continue
            }

            # 1) Generate the C4 per-model plan
            Write-Host "[STEP 1] Generating C4 plan for $ds / $ml / $shortTag ..." -ForegroundColor DarkCyan
            python -m evaluation.prepare_conditions `
                --dataset $ds `
                --ollama-url $ollamaUrl `
                --ollama-model $ollamaTag `
                --llm-tag $planTag `
                --condition c4 `
                --user-context $ctxFile `
                --debug-dir "debug"

            if ($LASTEXITCODE -ne 0) {
                Write-Host "[FAIL] prepare_conditions for $ds/$ml/$shortTag" -ForegroundColor Red
                $failedSteps += "${ds}|${ml}|${shortTag}: prepare failed"
                continue
            }

            # 2) Run downstream evaluation with that plan
            Write-Host "[STEP 2] Evaluating $ds / $ml / $shortTag ..." -ForegroundColor DarkCyan
            python -m evaluation.run_experiments `
                --datasets $ds `
                --llm-tag $planTag `
                --condition C4

            if ($LASTEXITCODE -ne 0) {
                Write-Host "[FAIL] run_experiments for $ds/$ml/$shortTag" -ForegroundColor Red
                $failedSteps += "${ds}|${ml}|${shortTag}: evaluate failed"
                continue
            }

            $successSteps++
            Write-Host "[OK] $ds | $ml | $shortTag" -ForegroundColor Green
        }
    }

    # Plan quality for each per-model plan
    Write-Host ""
    Write-Host "[STEP 3] Plan quality for $shortTag combos ..." -ForegroundColor DarkCyan
    foreach ($ds2 in $datasets.Keys) {
        foreach ($ml2 in $datasets[$ds2]) {
            $pqTag = "${shortTag}_${ml2}"
            python -m evaluation.plan_quality --llm-model $pqTag --condition c4 --datasets $ds2
            if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] plan_quality for $pqTag/$ds2" -ForegroundColor Magenta }
        }
    }
}

# Consolidate everything
Write-Host ""
Write-Host "===== CONSOLIDATING RESULTS =====" -ForegroundColor Cyan
python -m evaluation.consolidate_results

Write-Host ""
Write-Host "===== PHASE B COMPLETE =====" -ForegroundColor Green
Write-Host "  Total combos: $totalSteps"
Write-Host "  Success: $successSteps"
Write-Host "  Failed:  $($failedSteps.Count)"
if ($failedSteps.Count -gt 0) {
    Write-Host "  Failures:" -ForegroundColor Red
    foreach ($f in $failedSteps) {
        Write-Host "    - $f" -ForegroundColor Red
    }
}
Write-Host ""
Write-Host "Results in evaluation/outputs/" -ForegroundColor Green
Write-Host "  - MASTER_RESULTS_TABLE.csv  (now includes Phase B)"
Write-Host "  - MASTER_PLAN_QUALITY.csv   (now includes Phase B)"
