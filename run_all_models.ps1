# run_all_models.ps1
# ----------------------------------------------------------------------
# Runs the full Phase A benchmark over the 7 LLM models.
# Phase A: 3 primary datasets (adult, diabetes, student).
# Phase B: enable the second block at the bottom when you are ready.
#
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File .\run_all_models.ps1
#
# Pre-conditions:
#   - Ollama is running:   ollama serve
#   - All 7 models pulled (see comments below for ollama pull commands)
#   - Python venv active:  .\.venv\Scripts\Activate.ps1
#   - c0_raw.csv generated for each dataset (data/scripts/prepare_datasets.py)
# ----------------------------------------------------------------------

$ErrorActionPreference = "Stop"

# Each entry: ollama tag, plus a short tag used in filenames.
# Adjust the tag on the right ONLY if a model isn't available locally.
$models = @(
    @{ ollama = "qwen2.5:3b";       tag = "qwen2.5_3b"      },
    @{ ollama = "llama3.2:3b";      tag = "llama3.2_3b"     },
    @{ ollama = "mistral:7b";       tag = "mistral_7b"      },
    @{ ollama = "qwen2.5:7b";       tag = "qwen2.5_7b"      },
    @{ ollama = "llama3.1:8b";      tag = "llama3.1_8b"     },
    @{ ollama = "gemma2:9b";        tag = "gemma2_9b"       },
    @{ ollama = "mistral-nemo:12b"; tag = "mistral_nemo_12b"},
    @{ ollama = "qwen2.5:14b";      tag = "qwen2.5_14b"     }
    @{ ollama = "qwen3:8b";    tag = "qwen3_8b"   },
    @{ ollama = "qwen3:14b";   tag = "qwen3_14b"  },
    @{ ollama = "gemma3:12b";  tag = "gemma3_12b" },
    @{ ollama = "phi4:14b";    tag = "phi4_14b"   },
    @{ ollama = "gpt-oss:20b"; tag = "gpt_oss_20b"}
)

$datasets = "adult,diabetes,student,life_expectancy"
$ollamaUrl = "http://localhost:11434"

Write-Host "===== PHASE A: 8 models x 4 datasets (3 classification + 1 regression) =====" -ForegroundColor Cyan

foreach ($m in $models) {
    $ollamaTag = $m.ollama
    $shortTag  = $m.tag
    Write-Host ""
    Write-Host "----- $ollamaTag ($shortTag) -----" -ForegroundColor Yellow

    # 1) Generate the LLM plans (per-dataset). Saves c2_llm_plan_<tag>.json under provenance/.
    python -m evaluation.prepare_conditions `
        --dataset all `
        --ollama-url $ollamaUrl `
        --ollama-model $ollamaTag `
        --llm-tag $shortTag `
        --debug-dir "debug"
    if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] prepare_conditions for $shortTag" -ForegroundColor Red; continue }

    # 2) Run downstream evaluation with that LLM's plan in the cleaning pipeline.
    python -m evaluation.run_experiments `
        --datasets $datasets `
        --llm-tag $shortTag
    if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] run_experiments for $shortTag" -ForegroundColor Red; continue }

    # 3) Plan-quality metrics (parses provenance + plan).
    python -m evaluation.plan_quality --llm-model $shortTag
    if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] plan_quality for $shortTag" -ForegroundColor Magenta }
}

# 4) Consolidate everything into the master tables.
python -m evaluation.consolidate_results
Write-Host ""
Write-Host "===== PHASE A DONE =====" -ForegroundColor Green
Write-Host "Master tables in evaluation/outputs/:"
Write-Host "  - MASTER_RESULTS_TABLE.csv  (downstream metrics x model)"
Write-Host "  - MASTER_PLAN_QUALITY.csv   (plan-quality metrics x model)"

# ----------------------------------------------------------------------
# PHASE B (uncomment after picking the best model from Phase A)
# Replace 'qwen2.5_7b' with the winner's short tag.
# ----------------------------------------------------------------------
# $bestOllama = "qwen2.5:7b"
# $bestTag    = "qwen2.5_7b"
# $phaseBDatasets = "house_prices,heart,bank"
#
# python -m evaluation.prepare_conditions `
#     --dataset all `
#     --ollama-url $ollamaUrl `
#     --ollama-model $bestOllama `
#     --llm-tag $bestTag `
#     --debug-dir "debug"
#
# python -m evaluation.run_experiments `
#     --datasets $phaseBDatasets `
#     --llm-tag $bestTag
#
# python -m evaluation.plan_quality --llm-model $bestTag
# python -m evaluation.consolidate_results
