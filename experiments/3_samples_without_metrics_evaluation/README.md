# Run: Qualitative Samples – Exclusive Multimodal Flips (Prompt testing)

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
This folder contains only the qualitative examples where the model's prediction flipped exclusively under multimodal corruption — that is, the classification changed only when both modalities (text and image) were jointly perturbed.

Unlike the earlier run, this execution was carried out with a slightly modified text corruption prompt, which likely influenced the linguistic style and semantic distribution of the generated paraphrases.  
Although no quantitative metrics were computed for this run, the resulting samples illustrate clear cases of cross-modal vulnerability amplification, where neither unimodal attack alone was sufficient to fool the model.
