# Run: Pipeline Execution – Unfiltered Text Similarity

**Date:** November 2025  

---

## Key Parameters
| Parameter | Value |
|------------|--------|
| **PGD iterations** | 25 |
| **PGD ε (epsilon)** | 3 / 255 |
| **Batch size** | 8 |
| **Text tokens (max)** | 128 |

---

## Discussion
In this run, the pipeline was executed **without applying any text similarity threshold** before accepting the corrupted text.  
As a result, some rewritten samples exhibited low semantic coherence or mild hallucinations, but the experiment produced **8 cases** where the model’s prediction flipped **exclusively under multimodal corruption**.

While overall attack performance metrics were **lower** compared to the filtered version (where text similarity ≥ 0.5), this execution highlighted the **nonlinear nature** of multimodal vulnerabilities — showing that even suboptimal text perturbations can catalyze classification failures when combined with visual noise.

This run provides an important intermediate baseline before the introduction of semantic filtering.
