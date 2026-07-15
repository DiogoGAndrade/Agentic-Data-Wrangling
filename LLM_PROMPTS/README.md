# LLM_PROMPTS — Guia de utilização

Esta pasta contém tudo o que precisas para comparar ChatGPT, Gemini, Claude e Copilot com o teu sistema C4.

---

## Estrutura

```
LLM_PROMPTS/
  ChatGPT/
    PROMPT_ADULT.txt       ← copia tudo isto e cola no chat
    PROMPT_DIABETES.txt
    PROMPT_BANK.txt
    PROMPT_HEART.txt
    RESPOSTAS/
      plan_chatgpt_adult.json    ← cola aqui o JSON que o LLM devolver
      plan_chatgpt_diabetes.json
      plan_chatgpt_bank.json
      plan_chatgpt_heart.json
  Gemini/   (igual)
  Claude/   (igual)
  Copilot/  (igual)
```

---

## Passo a passo (repetes para cada LLM e cada dataset)

### 1. Abrir o ficheiro de prompt
Abre, por exemplo, `ChatGPT/PROMPT_ADULT.txt` com o Notepad ou VS Code.  
Selecciona tudo (Ctrl+A) e copia (Ctrl+C).

### 2. Colar no chat do LLM
- **ChatGPT**: https://chat.openai.com — novo chat, cola, envia
- **Gemini**: https://gemini.google.com — novo chat, cola, envia
- **Claude**: https://claude.ai — novo chat, cola, envia
- **Copilot**: https://copilot.microsoft.com — novo chat, cola, envia

### 3. Receber o JSON
O LLM vai devolver um JSON com esta estrutura:
```json
{
  "actions": [
    {
      "action": "handle_missing",
      "target_columns": ["workclass"],
      "params": {"strategy": "most_frequent"}
    },
    ...
  ]
}
```

Se o LLM devolver o JSON dentro de bloco markdown (```json ... ```), remove as ``` e fica só com o JSON puro.

### 4. Guardar na pasta RESPOSTAS
Abre o ficheiro placeholder correspondente em `RESPOSTAS/` e substitui o conteúdo pelo JSON recebido.  
Exemplo: `ChatGPT/RESPOSTAS/plan_chatgpt_adult.json`

### 5. Copiar para a pasta do pipeline
Depois de teres todos os JSONs de um LLM, copia-os para:
```
evaluation/cloud_llm_comparator/
```
Mantendo o nome `plan_<llm>_<dataset>.json`.

### 6. Avaliar (1 comando no PowerShell)
```powershell
cd "C:\Users\dcsga\OneDrive\Ambiente de Trabalho\Tese\Projeto"
.\.venv\Scripts\Activate.ps1

# Um LLM de cada vez:
python -m evaluation.apply_cloud_llm_plan --llm chatgpt --dataset all
python -m evaluation.apply_cloud_llm_plan --llm gemini --dataset all
python -m evaluation.apply_cloud_llm_plan --llm claude --dataset all
python -m evaluation.apply_cloud_llm_plan --llm copilot --dataset all

# Consolidar no MASTER:
python -m evaluation.consolidate_results
```

---

## Dicas por LLM

**ChatGPT (GPT-4o)**
- Geralmente devolve JSON limpo se o prompt pedir "Return ONLY valid JSON"
- Se começar com texto explicativo, diz "Only the JSON, no explanation"

**Gemini (1.5 Pro / 2.0 Flash)**
- Pode devolver em markdown (```json ... ```) — remove as aspas
- Experimente o Gemini Advanced para resultados mais ricos

**Claude (Sonnet/Opus)**
- Tende a ser muito rigoroso com o formato — JSON geralmente limpo
- Se o contexto for muito grande (diabetes: 47 colunas), pode truncar; nesse caso divide o contexto em 2 mensagens

**Copilot (Microsoft)**
- Usa GPT-4o por baixo mas com sistema prompt mais conservador
- Pode recusar-se a gerar código/JSON em alguns contextos — reformula como "generate a structured plan"

---

## O que fazer se o JSON tiver erros

Erros comuns:
- Nome de acção errado: `"impute"` em vez de `"handle_missing"` → corrige manualmente
- Parâmetros em falta: adiciona `"params": {}` se estiver vazio
- Colunas que não existem no dataset: remove essa acção

As acções válidas são (exactamente estas strings):
`handle_missing`, `encode_categorical_per_column`, `clip_outliers`,
`transform_numeric_skewed`, `add_missing_indicators`, `group_rare_categories`,
`select_features`, `bin_numeric`

---

## Prioridade recomendada

Faz por esta ordem (do mais informativo para a tese ao menos):

1. **adult** — missing values + native_country com 41 categorias raras
2. **bank** — pdays=999 semântica, education ordinal (aqui o C4 tem WIN)
3. **diabetes** — 47 colunas, 95% NA em max_glu_serum, 700+ ICD codes (WIN do C4)
4. **heart** — pequeno (303 linhas), rápido de avaliar

Se tiveres pouco tempo, faz só **adult + bank** para os 4 LLMs — já é suficiente para o argumento da tese.
