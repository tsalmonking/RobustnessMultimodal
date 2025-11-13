# Run: Exploratory Prompt and Token-Length Tests – Unstable Text Corruptions

**Date:** November 2025  

---

## Key Parameters
| Parameter | Value |
|------------|--------|
| **PGD iterations** | 20 |
| **PGD ε (epsilon)** | 4 / 255 |
| **Batch size** | *Variable / Not logged* |
| **Text tokens (max)** | *Variable / Not logged* |

---

## Discussion
This experiment was conducted as part of the prompt and token-length exploration phase, aimed at understanding how text corruption quality and model behavior change under different input configurations.

The text perturbations in this run were highly unstable.

Despite this, the experiment produced a large number of cases where the model flipped its prediction only under multimodal perturbation, confirming once again that the fusion mechanism is highly sensitive to coordinated cross-modal noise.

No quantitative metrics were collected for this run, as its primary purpose was qualitative diagnosis of prompt robustness and multimodal sensitivity.