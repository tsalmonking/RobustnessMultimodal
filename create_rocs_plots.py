import subprocess

if __name__ == "__main__":
    # Clean rocs for multimodal configurations
    subprocess.run(["python", "metrics.py", "--type", "clean", "--modality", "feature-fusion", "--roc-set", "multimodal-clean"])
    subprocess.run(["python", "metrics.py", "--type", "clean", "--modality", "late-fusion", "--mode", "mean", "--roc-set", "multimodal-clean"])
    subprocess.run(["python", "metrics.py", "--type", "clean", "--modality", "late-fusion", "--mode", "min", "--roc-set", "multimodal-clean"])
    subprocess.run(["python", "metrics.py", "--type", "clean", "--modality", "late-fusion", "--mode", "max", "--roc-set", "multimodal-clean"])

    # Perturbed rocs for multimodal configurations
    subprocess.run(["python", "metrics.py", "--type", "perturbed", "--modality", "feature-fusion", "--roc-set", "multimodal-perturbed"])
    subprocess.run(["python", "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "mean", "--roc-set", "multimodal-perturbed"])
    subprocess.run(["python", "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "min", "--roc-set", "multimodal-perturbed"])
    subprocess.run(["python", "metrics.py", "--type", "perturbed", "--modality", "late-fusion", "--mode", "max", "--roc-set", "multimodal-perturbed"])
    subprocess.run(["python", "metrics.py", "--type", "perturbed", "--modality", "feature-fusion", "--mode", "text-perturbed", "--roc-set", "multimodal-perturbed"])
    subprocess.run(["python", "metrics.py", "--type", "perturbed", "--modality", "feature-fusion", "--mode", "image-perturbed", "--roc-set", "multimodal-perturbed"])

    
    # Clean rocs for unimodal configurations
    subprocess.run(["python", "metrics.py", "--type", "clean", "--modality", "text", "--roc-set", "unimodal-clean"])
    subprocess.run(["python", "metrics.py", "--type", "clean", "--modality", "image", "--roc-set", "unimodal-clean"])

    # Perturbed rocs for unimodal configurations
    subprocess.run(["python", "metrics.py", "--type", "perturbed", "--modality", "text", "--roc-set", "unimodal-perturbed"])
    subprocess.run(["python", "metrics.py", "--type", "perturbed", "--modality", "image", "--roc-set", "unimodal-perturbed"])
    subprocess.run(["python", "metrics.py", "--type", "perturbed", "--modality", "feature-fusion", "--mode", "text-perturbed", "--roc-set", "unimodal-perturbed"])
    subprocess.run(["python", "metrics.py", "--type", "perturbed", "--modality", "feature-fusion", "--mode", "image-perturbed", "--roc-set", "unimodal-perturbed"])
