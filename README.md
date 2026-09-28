# Authorization Behavior Under Competing Context

A cross-model behavioral benchmark for studying how language models make authorization decisions when explicit policy conflicts with competing contextual cues.

This project began as a single-model behavioral-forensics study during a **BlueDot Impact Technical AI Safety Project Sprint** and was later expanded into a seven-model benchmark.

## Research question

> **What controls a language model's authorization decision when explicit policy conflicts with competing contextual information?**

The benchmark is designed to separate clean policy-following ability from robustness under conflicting context. The authorization environment is an experimental harness: its purpose is to make it cheap to vary one contextual factor at a time while holding the underlying policy decision fixed.

The benchmark is **not intended as a model leaderboard**. It is intended to characterize behavioral profiles such as:

- baseline authorization competence,
- susceptibility to different kinds of contextual conflict,
- direction of decision changes,
- matched ALLOW / DENY asymmetries,
- exact output-protocol adherence,
- and scenario-level susceptibility across models.

## Why this matters

Language models used as agents may operate in environments containing:

- permissions,
- organizational roles,
- policies,
- identity and user state,
- tool outputs,
- workflow state,
- operational pressure,
- and informal instructions.

A model can apply an authorization rule correctly in a clean prompt but make a different decision when additional context conflicts with that rule.

For agent systems, this is relevant to:

- behavioral evaluation,
- oversight,
- safe tool use,
- permission-aware reasoning,
- authorization controls,
- and robustness to conflicting instructions.

## Benchmark design

The benchmark contains **100 authorization scenarios** organized into **50 matched policy families**.

Each family contains:

- one ground-truth `ALLOW` scenario,
- one ground-truth `DENY` scenario,
- the same policy,
- the same user role,
- and a different requested action.

The scenarios span ten domains:

- databases,
- software deployment,
- cloud infrastructure,
- finance,
- human resources,
- security,
- operations,
- data privacy,
- documents and records,
- identity and access management.

Each scenario is evaluated under **11 conditions**:

| Condition | Description |
|---|---|
| `clean` | Policy, role, and action with no competing context |
| `peer_conflict` | A colleague recommends the opposite decision |
| `manager_conflict` | A manager recommends the opposite decision |
| `executive_conflict` | A senior executive recommends the opposite decision |
| `urgency_conflict` | Opposite recommendation combined with operational urgency |
| `peer_before_policy` | Peer conflict inserted before the policy |
| `peer_after_policy` | Peer conflict inserted after the policy |
| `emphatic_conflict` | Opposite recommendation presented with strongly salient wording |
| `user_state_conflict` | Competing claim about whether the requester is authorized |
| `exception_conflict` | Opposite decision framed as a one-time exception |
| `workaround_conflict` | Opposite decision framed as a temporary workaround |

With seven models, the current benchmark contains:

```text
7 models × 100 scenarios × 11 conditions = 7,700 evaluations
```

## Models

The current local benchmark panel is:

| Benchmark ID | Ollama model |
|---|---|
| `qwen3_4b` | `qwen3:4b-instruct-2507-q4_K_M` |
| `gemma3_4b` | `gemma3:4b` |
| `llama3_2_3b` | `llama3.2:3b` |
| `granite3_3_8b` | `granite3.3:8b` |
| `llama3_1_8b` | `llama3.1:8b` |
| `mistral_7b_instruct` | `mistral:instruct` |
| `phi4_14b` | `phi4:14b` |

Inference is run locally through Ollama at temperature `0`.

## Output interpretation

The original benchmark required the model to return exactly:

```text
ALLOW
```

or:

```text
DENY
```

Cross-model evaluation revealed that several models reached a clear authorization decision but added explanations. Treating these responses as authorization failures would mix two different capabilities.

The analysis therefore separates:

### Strict output validity

The raw response is exactly `ALLOW` or `DENY`.

### Semantic decision validity

A conservative parser can recover one unambiguous `ALLOW` or `DENY` decision from the raw response.

Behavioral analyses use the semantic decision. Strict format adherence is retained as a separate operational metric because exact interface compliance also matters for tool-using agents.

## Headline results

Semantic clean accuracy is high across all seven models, but contextual susceptibility varies substantially.

| Model | Clean accuracy | User-state flip rate | Emphatic flip rate | Strict output validity |
|---|---:|---:|---:|---:|
| Granite 3.3 8B | 99% | 3.0% | 11.0% | 46.5% |
| Llama 3.1 8B | 98% | 18.0% | 4.0% | 21.8% |
| Phi-4 14B | 98% | 13.0% | 4.3% | 0.0% |
| Qwen3 4B | 98% | 23.0% | 13.0% | 100% |
| Mistral 7B Instruct | 96% | 60.6% | 53.5% | 35.5% |
| Gemma 3 4B | 93% | 66.0% | 59.0% | 100% |
| Llama 3.2 3B | 92% | 55.0% | 38.0% | 100% |

### Main observations

1. **Clean authorization competence does not predict contextual robustness.**  
   All seven models achieve high clean semantic accuracy, while their susceptibility to conflicting context differs dramatically.

2. **User-state conflict is the most consistently disruptive intervention.**  
   It produces the largest flip rate for six of the seven tested models.

3. **Models have distinct behavioral profiles.**  
   Some models are selectively sensitive to user-state information; others are broadly affected by authority, urgency, emphatic language, exceptions, and workarounds.

4. **Direction matters.**  
   Most models show more `ALLOW → DENY` than `DENY → ALLOW` changes under user-state conflict, although the pattern is not universal.

5. **Matched policy families reveal an asymmetry.**  
   Several models have substantially more families in which only the ALLOW member is susceptible than families in which only the DENY member is susceptible.

6. **Some scenarios are susceptible across multiple model families.**  
   This suggests that part of the effect is associated with scenario structure rather than model identity alone.

7. **Protocol adherence and authorization reasoning are different properties.**  
   Some models achieve high semantic accuracy while frequently ignoring the requested exact binary output format.

## From single-model study to benchmark

The project originally evaluated only Qwen3-4B-Instruct.

In that first 1,100-decision sweep:

- clean accuracy was 98%,
- most ordinary contextual interventions caused very few changes,
- emphatic conflict caused 13 flips,
- user-state conflict caused 23 flips.

A follow-up reran the 23 user-state-sensitive scenarios across all interventions. User-state conflict reproduced on 22 of 23 cases, while most other interventions changed almost none of them.

This suggested that the effect was not simply general prompt fragility.

The project was then expanded into the current seven-model benchmark to test whether the same behavioral patterns appear across model families.

## Repository structure

A typical project layout is:

```text
authorization-behavior/
├── config/
│   └── models.yaml
├── data/
│   ├── scenarios.yaml
│   └── selections/
├── runs/
├── src/
│   └── authorization_behavior/
│       ├── main.py
│       ├── model_config.py
│       ├── models.py
│       ├── parsing.py
│       ├── prompts.py
│       ├── interventions.py
│       ├── runner.py
│       ├── runs.py
│       ├── scenarios.py
│       ├── analysis.py
│       └── benchmark_analysis.py
├── notebooks/
├── pyproject.toml
└── uv.lock
```

## Running the benchmark

### Prerequisites

- Python
- [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)

Install the project environment:

```bash
uv sync
```

Make sure the models listed in `config/models.yaml` are available in Ollama.

For example:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull gemma3:4b
ollama pull llama3.2:3b
```

### Smoke test one model

```bash
uv run python -m authorization_behavior.main \
    --model qwen3_4b \
    --limit 5
```

### Run one complete model

```bash
uv run python -m authorization_behavior.main \
    --model qwen3_4b
```

A complete run contains:

```text
100 scenarios × 11 conditions = 1,100 model calls
```

### Run all configured models

```bash
uv run python -m authorization_behavior.main
```

### Analyze completed benchmark runs

```bash
uv run python -m authorization_behavior.benchmark_analysis
```

The benchmark analysis:

- ignores incomplete smoke-test runs,
- keeps one complete run per model,
- derives semantic decisions from stored raw responses,
- computes clean accuracy,
- computes condition accuracy,
- computes flip rates relative to clean,
- measures `ALLOW → DENY` and `DENY → ALLOW` transitions,
- measures degradation of clean-correct decisions,
- analyzes matched policy families,
- and identifies scenarios that are susceptible across models.

## Reproducibility

Each evaluation stores information including:

- model ID,
- model name,
- scenario ID,
- scenario metadata,
- condition,
- rendered prompt,
- raw response,
- parsed decision,
- ground truth,
- correctness,
- and latency.

Run results are persisted as Parquet files under `runs/`.

Saving raw model outputs makes it possible to change the parsing and analysis logic without rerunning inference.

No LLM judge is used.

## Demo notebook

The demo notebook provides a compact walkthrough of the benchmark, including:

- examples of all interventions,
- model × intervention flip-rate heatmaps,
- semantic accuracy heatmaps,
- user-state directionality,
- strict output versus semantic validity,
- degradation of clean-correct decisions,
- matched-family asymmetry,
- and cross-model scenario susceptibility.

Start JupyterLab with:

```bash
uv run jupyter lab
```

## Limitations

- The scenarios are synthetic and deliberately simple.
- The benchmark currently evaluates locally served models rather than frontier hosted systems.
- Models differ in size, training, instruction tuning, chat templates, and local quantization.
- Temperature `0` characterizes deterministic behavior rather than sampling variability.
- The user-state intervention does not currently specify provenance or source reliability.
- Some interventions, especially emphatic conflict, modify several surface features simultaneously.
- The matched-family and cross-model analyses are exploratory.
- The current benchmark uses a single scenario set rather than separate development and held-out evaluation sets.
- The study is behavioral and does not identify the internal mechanism producing the observed effects.

## Next experiment

The highest-value follow-up is a **provenance-controlled user-state experiment**.

Hold the contradictory authorization claim fixed while varying where it comes from:

- an unverified note,
- a colleague,
- a manager,
- an authoritative identity service,
- or structured system state explicitly designated as authoritative.

This can help distinguish between competing explanations such as:

- **source ambiguity** — the model does not know which representation of user state should be trusted;
- **asymmetric conflict resolution** — negative authorization evidence receives more weight than positive evidence even when provenance is explicit;
- **conservative conflict resolution** — contradictory context induces a model-specific tendency toward denial.

## Project status

This repository contains work from a **BlueDot Impact Technical AI Safety Project Sprint, September 2026**.

The current benchmark should be viewed as a **behavioral map and hypothesis generator** rather than a deployment-grade authorization test or a definitive model ranking.

## Author

**Stefan Höglund**

