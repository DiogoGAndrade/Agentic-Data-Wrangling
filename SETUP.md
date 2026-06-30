# Setup guide

This guide takes you from a clean machine to a running app, assuming no prior context
about this project. It covers Windows, macOS, and Linux; commands differ where noted.

## 1. Prerequisites

- **Python 3.10 or newer**. Check with `python --version` (or `python3 --version` on
  macOS/Linux). If you do not have it, download from
  [python.org/downloads](https://www.python.org/downloads/).
- **Git**, to clone this repository.
- **Ollama**, to run the local LLMs. Download from [ollama.com](https://ollama.com) and
  install it. On Windows and macOS it runs as a background service after installation;
  on Linux, start it with `ollama serve` in a terminal (leave that terminal open, or run
  it as a systemd service per the Ollama docs).
- About 15-20 GB of free disk space if you intend to pull every model used in the
  thesis; a single 3B model is under 3 GB if you just want to try the app.

## 2. Clone and install

```bash
git clone <this-repository-url>
cd <repository-folder>

# Optional but recommended: an isolated virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Pull at least one model

The app and the evaluation scripts call Ollama's local HTTP API, so a model must be
pulled before you use it. From a terminal, with Ollama running:

```bash
ollama pull qwen2.5:3b-instruct
```

This is the smallest model the thesis evaluates and the fastest to download and run.
The full set used in the study, if you want to reproduce the benchmark exactly, is:

```bash
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:14b-instruct
ollama pull llama3.2:3b-instruct
ollama pull llama3.1:8b-instruct
ollama pull mistral:7b-instruct
ollama pull mistral-nemo:12b-instruct
ollama pull gemma2:9b-instruct
```

Confirm Ollama is reachable:

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON list including the model(s) you just pulled.

## 4. Run the app

```bash
streamlit run App/main.py
```

Streamlit prints a local URL (usually `http://localhost:8501`); open it in a browser.
From there:

1. Upload a CSV.
2. Confirm or correct the auto-detected target column.
3. Pick an installed model from the dropdown (models you have pulled show a checkmark).
4. Click "Propose plan" to see the LLM's cleaning plan before anything runs, or
   "Auto-clean" to apply it directly.
5. Download the cleaned dataset and the audit log from the results panel.

If the model dropdown shows nothing as installed, Ollama is either not running or not
reachable at `http://localhost:11434`; check that the Ollama application or service is
active.

## 5. Run the tests

```bash
pytest
```

This runs the unit tests for the cleaning-action executors in `tests/` and does not
require Ollama.

## 6. Reproduce the evaluation (optional, advanced)

The full benchmark used in the thesis downloads several public datasets, calls eight
local LLMs across five experimental conditions, and runs five-fold cross-validated
downstream evaluation. It takes several hours on a single machine, depending on
hardware (the thesis reports roughly 3 minutes per plan for the smallest model and
roughly 18 minutes for the largest, per dataset).

1. Place the raw datasets under `data/raw/<dataset_name>/` (see
   `evaluation/README.md` for the exact source and expected filenames for each of the
   nine datasets; they are all public).
2. Pull every model listed in `run_all_models.ps1` (Section 3 above lists them).
3. From the project root, on Windows PowerShell:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run_all_models.ps1
   ```

   On macOS/Linux, run the equivalent steps directly with Python; open
   `run_all_models.ps1` to see the exact sequence of `evaluation/*.py` scripts it calls
   for each model, and run them with `python3` instead.

4. Results land in `evaluation/outputs/MASTER_RESULTS_TABLE.csv` (downstream metrics)
   and `evaluation/outputs/MASTER_PLAN_QUALITY.csv` (plan-quality metrics: actions
   proposed, applied, rejected, and latency).

To add or remove models from the benchmark, edit the `$models` list at the top of
`run_all_models.ps1`. To change the dataset list, edit `engine/config.py`
(`TASK_TYPE`) and the corresponding entries in `evaluation/prepare_conditions.py`.

## Troubleshooting

- **"Could not reach Ollama"**: confirm the Ollama application/service is running and
  that `curl http://localhost:11434/api/tags` returns a response.
- **A model is missing from the dropdown**: it has to be pulled first
  (`ollama pull <tag>`); the app only lists models Ollama already has locally, plus a
  recommended set it can prompt you to pull.
- **`pip install` fails on `python-docx` or a scientific package**: make sure you are
  on Python 3.10+ and, on Windows, that you have the latest pip
  (`python -m pip install --upgrade pip`).
- **Slow plan generation**: the 14B-parameter models are roughly six times slower than
  the 3B ones on the same hardware for no measurable gain in downstream quality once
  guardrails are enforced (see the thesis, Section 5.1.4); the 3B model is the
  recommended default for everyday use.
