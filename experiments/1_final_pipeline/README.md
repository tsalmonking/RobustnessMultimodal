# Run: Final Pipeline Execution – No Exclusive Multimodal Flips

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
The absence of exclusively multimodal flips does not indicate robustness but rather a limitation of the current batch evaluation process, where fine-grained sample-level tracking was affected by batching dynamics.
