# MMM from Scratch — Key Concepts (So Far)

This document summarizes the key concepts, decisions, and mental models introduced so far in the project.

Its goal is to serve as a **reference guide** while progressing through the MMM implementation.

---

## 1. What Marketing Mix Modeling (MMM) is (and is not)

### What MMM is
- A **top-down, observational model** that relates marketing inputs (usually spend) to a business KPI over time.
- A tool for **strategic decision support**, not ground-truth measurement.
- Best suited for **medium- to long-term planning** and cross-channel allocation.

### What MMM is NOT
- A causal inference method by itself.
- A campaign- or keyword-level optimizer.
- A replacement for experiments or incrementality testing.

---

## 2. High-level vs Low-level budget allocation

### High-level allocation
Decisions across channels, such as:
- How to split budget between Search, Social, Display, etc.
- What happens if we shift +20% spend from one channel to another.
- Quarterly or monthly planning.

This is the **primary use case of MMM**.

### Low-level allocation
Decisions within channels, such as:
- Campaign / ad set / creative allocation.
- Daily or weekly pacing.
- Keyword-level optimization.

MMM alone is **not sufficient** for this.
Low-level allocation requires an additional layer:
- Experiments (geo-lift, incrementality tests)
- Within-channel models or rules
- Operational constraints

MMM should be seen as a **top-down allocator**, complemented by bottom-up tools.

---

## 3. Target (KPI) definition

Key principles:
- One primary KPI per model.
- Must be stable, interpretable, and consistently measured.
- Weekly aggregation is standard for MMM.

In this project:
- KPI: purchase count (conversions)
- Raw data: daily
- Modeling data: weekly

Revenue-based KPIs can be explored later once the pipeline is stable.

---

## 4. Why MMM uses weekly aggregation

Reasons:
- Reduces daily noise.
- Smooths sparse channel activity.
- Improves model stability.
- Aligns with common MMM practice.

Daily MMM is possible but significantly harder to stabilize and interpret.

---

## 5. Carryover effects (Adstock)

### Concept
Marketing impact may persist over time.
Spend today can affect outcomes in future periods.

### Why it matters
Without carryover:
- Upper-funnel channels appear ineffective.
- Lower-funnel channels absorb too much credit.

Adstock models this delayed effect explicitly.

---

## 6. Diminishing returns (Saturation)

### Concept
The marginal impact of spend decreases as spend increases.

### Why it matters
Without saturation:
- The model assumes linear scaling.
- Budget recommendations become unrealistic.
- One channel may dominate allocation unfairly.

Saturation constrains the response curve.

---

## 7. Linear models in MMM

MMM often uses linear regression **after transformations**:
- Adstock handles time dynamics.
- Saturation handles non-linearity.

This preserves:
- Interpretability
- Debuggability
- Clear baselines

---

## 8. Correlation vs causation

MMM estimates **associations**, not causal effects.

Implications:
- Coefficients reflect historical patterns.
- External factors can bias estimates.
- Results must be interpreted with caution.

Causality requires:
- Experiments
- Quasi-experimental designs
- External validation

MMM supports decisions; it does not prove truth.

---

## 9. Parameter stability assumption

MMM assumes effects are reasonably stable over the modeling window.

Violations occur when:
- Product changes
- Pricing changes
- Platform algorithms change
- Business model shifts

Mitigations:
- Temporal validation
- Window segmentation
- Sensitivity analysis

---

## 10. Key risks and limitations

### Omitted variable bias
Missing drivers (promotions, pricing, PR) can distort attribution.

### Multicollinearity
Channels moving together are hard to disentangle.

### Functional misspecification
Wrong adstock or saturation choices bias results.

### Short-term shocks
Unexpected events can mislead the model.

### Sensitivity to time window
Results may change with different periods or validation strategies.

Frameworks mitigate these issues but do not eliminate them.

---

## 11. Frameworks vs from-scratch MMM

Modern frameworks:
- Automate good practices
- Explore parameter space efficiently
- Improve validation and stability

They do NOT:
- Create causality
- Fix bad data
- Replace domain understanding

Building MMM from scratch builds the intuition needed to use frameworks responsibly.

---

## 12. Decision-driven notebooks

Key communication principles:
- Notebooks document **decisions**, not everything tried.
- Each section should end with:
  - Decision
  - Rationale
- Assumptions and limitations are explicit.
- Outputs that do not change decisions require minimal commentary.

This style improves:
- Readability
- Auditability
- Long-term project maintainability

---

## 13. Where this fits in the project

So far, we have:
- Defined the business question and scope
- Selected a single coherent time series
- Chosen a stable KPI
- Agreed on modeling assumptions and risks

Next step:
- Build a baseline MMM to establish a reference point.
