# Notebook Style Guide

This document defines how notebooks are written and communicated in this project.

The goal is to make notebooks **decision-driven**, readable, and reusable, rather than verbose execution logs.

---

## Core principle

> **A notebook documents decisions, not everything that was tried.**

If an output does not change a decision, it does not require a long explanation.

---

## Notebook philosophy

Notebooks in this project are:
- Decision-oriented
- Explicit about assumptions
- Focused on modeling choices
- Designed to be read top-to-bottom

They are **not**:
- Exhaustive EDA logs
- Step-by-step code tutorials
- Narrative blog posts

---

## Standard notebook structure

Each notebook should clearly answer four questions:

1. What decision is this notebook addressing?
2. What evidence is used?
3. What decision is taken?
4. What remains out of scope?

---

## Recommended header template

Each notebook should start with:

```markdown
# <NN> — <Notebook title>

## Objective
What decision or question is this notebook addressing?

## Inputs
What data, assumptions, or previous decisions does this notebook rely on?

## Output
What concrete decision or artifact should exist at the end of this notebook?
```

---
## Section-level communication pattern

Each logical section in a notebook should follow this structure:

### 1. Context (Markdown)
Explain *why* this step or check matters for the model.

### 2. Evidence (Python)
Code that produces the evidence (tables, plots, statistics).

### 3. Interpretation (Markdown)
Explain what the evidence suggests or rules out.

### 4. Decision (Markdown)
State clearly what decision is taken.

### 5. Rationale (Markdown)
Explain **why this decision was chosen over alternatives**.

This section should answer:
- Why is this preferable?
- What trade-offs are accepted?
- What assumptions are being made?

If no real decision is taken, keep this section short or omit it.

### Example

**Decision**  
Aggregate daily data to weekly frequency.

**Rationale**  
Weekly aggregation reduces noise and channel sparsity while preserving medium-term marketing effects.
Daily modeling could be explored later if short-term effects become critical.

---

## What to comment (and what not)

### Do comment
- Modeling assumptions
- Design decisions
- Trade-offs
- Risks and limitations
- Why one option is chosen over another

### Do NOT comment
- Obvious outputs (row counts, column lists)
- Code mechanics
- Library usage explanations

---

## "What would change my mind"

When possible, include statements like:

```markdown
This decision would be reconsidered if:
- Less than one year of data were available
- Spend were concentrated in a single channel
```

---

## Relationship with other artifacts

- **Notebooks** → decisions and evidence
- **Artifacts** → final outputs worth sharing
- **Writing / blog** → long-form explanations and reflections

Notebooks are not blogs.

---

## Final checklist before committing a notebook

- Is the objective of the notebook clear?
- Does each section end with a **decision**?
- Is the **rationale** for each decision explicit?
- Are assumptions and limitations stated?
- Could someone reproduce or extend the work by reading only Markdown?

If yes, the notebook is ready to commit.
