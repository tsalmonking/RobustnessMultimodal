# Run: PGD vs. DeepFool Comparison – Early Multimodal Flip Analysis

**Date:** October–November 2025  

---

## Key Parameters
| Parameter | Value |
|------------|--------|
| **PGD iterations** | 20 |
| **PGD ε (epsilon)** | 4 / 255 |
| **DeepFool iterations** | 200 |
| **DeepFool overshoot** | 0.0005 |
| **Batch size** | *Variable / Not logged* |
| **Text tokens (max)** | *Variable / Not logged* |

---

## Discussion
This experimental phase focused on comparing PGD and DeepFool as visual perturbation methods.  
The aim was to evaluate which attack provided a better balance between imperceptibility and classification effectiveness in a multimodal setting.

For both methods, the visual similarity (SSIM ) was recorded for each sample, allowing a more detailed inspection of how each attack affected perceptual.

The DeepFool variant was configured to also process perturbed text inputs, but this approach often produced degenerate results.  
PGD, by contrast, remained more stable and consistent across runs.

Despite the instability of DeepFool and the still unoptimized text prompt, this run generated multiple examples of exclusive multimodal flips, further supporting the hypothesis that cross-modal interactions amplify model fragility, even under imperfect attack conditions.

This phase was ultimately instrumental in confirming PGD as the preferred visual attack method for subsequent, more stable experiments.
