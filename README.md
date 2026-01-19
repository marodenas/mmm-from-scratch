# Marketing Mix Modeling from Scratch

This project explores **Marketing Mix Modeling (MMM)** from first principles, focusing on understanding **assumptions, limitations, and trade-offs** behind marketing attribution models.

The goal is not automation or production readiness, but **learning and decision support**.  
The project starts with a simple baseline MMM and progressively evolves towards comparisons with established frameworks (e.g. Robyn, Meridian).

---

## Core question

**How can I estimate the incremental impact of each marketing channel on a business KPI over time, while being explicit about the assumptions and limitations of the model?**

---

## Project philosophy

- Build models **from scratch before using frameworks**
- Prefer **clarity over complexity**
- Treat MMMs as **decision-support tools**, not ground truth
- Make assumptions explicit and actively challenge them
- Learn in public: notebooks + artifacts + writing

---

## Scope

### In scope
- MMM built from first principles
- Linear baseline models
- Manual implementation of:
  - adstock (carryover effects)
  - saturation (diminishing returns)
- Temporal validation (respecting time)
- Interpretation of results in a business context
- Comparison with external MMM frameworks (later stages)

### Out of scope (for now)
- Strong causal claims
- Fully automated MMM frameworks as a black box
- Production deployment
- Heavy Bayesian modeling
- Performance optimization

---

## Repository structure

```
notebooks/
  00_problem_context.ipynb
  01_data_audit_eda.ipynb
  02_baseline_model.ipynb
  03_adstock.ipynb
  04_saturation.ipynb
  05_validation.ipynb
  06_interpretation_next_steps.ipynb

  comparisons/
    10_robyn_setup_and_run.ipynb
    11_meridian_setup_and_run.ipynb
    12_compare_outputs.ipynb

src/
  ma_mmm_package/
    features/
    models/
    evaluation/
    optimization/
    pipelines/
    utils/

data/
  raw/
  interim/
  processed/
  external/

artifacts/
  figures/
  reports/
  models/
  runs/

writing/
  posts/
  notes.md
```

---

## Notebooks narrative

The notebooks are intentionally **sequential**.  
Each notebook represents a clear step in the learning process and can be read independently.

- `00` – Problem framing and assumptions
- `01` – Data audit and exploratory analysis
- `02` – Baseline MMM (no carryover, no saturation)
- `03` – Adstock (carryover effects)
- `04` – Saturation (diminishing returns)
- `05` – Temporal validation and diagnostics
- `06` – Interpretation, limitations, and next steps

The `comparisons/` folder introduces external frameworks once the fundamentals are well understood.

---

## Artifacts

The `artifacts/` folder contains **final outputs**, not exploratory work.

Artifacts include:
- Key figures used in posts or reports
- Final tables (e.g. channel contributions)
- Serialized models (optional)
- Results from different experimental runs

Rule of thumb:  
> If an output is worth showing or discussing, it belongs in `artifacts/`.

---

## Commit conventions

This project follows a **lightweight, intention-based commit style**.

### Commit format
```
<type>: <short description>
```

### Commit types used
- `chore` – project setup, structure, dependencies
- `data` – data loading, cleaning, schema definition
- `eda` – exploratory analysis
- `model` – modeling logic and feature engineering
- `eval` – validation, metrics, diagnostics
- `viz` – figures and visual outputs
- `docs` – README, notes, blog drafts
- `refactor` – internal code improvements
- `test` – tests

### Principles
- One commit = one intention
- Commits are made at the end of a session or logical step
- Commits describe *why*, not just *what*

Example commits:
```
model: add baseline linear regression MMM
model: implement geometric adstock transform
eval: add temporal train-test split
docs: draft post on adstock assumptions
```

---

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:
```bash
pip install -e ".[api]"
pip install -e ".[app]"
```

---

## Current status

- Phase 0: project scaffold and scope definition ✅
- Phase 1: data understanding and baseline MMM ⏳

---

## Notes

This repository is intentionally iterative.  
The code, notebooks, and conclusions are expected to evolve as understanding improves.

Learning > polish.
