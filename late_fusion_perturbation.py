import os

# Custom imports
from utils import create_late_fusion, create_late_fusion_parameters


if __name__ == "__main__":
    BASE = "data/Recovery/classification_results"

    CLEAN_BASE = os.path.join(BASE, "clean")
    PERT_BASE = os.path.join(BASE, "perturbed")

    # Input CSV
    clean_text_csv = os.path.join(CLEAN_BASE, "text", "results.csv")
    clean_image_csv = os.path.join(CLEAN_BASE, "image", "results.csv")

    per_text_csv = os.path.join(PERT_BASE, "text", "perturbed_results.csv")
    per_image_csv = os.path.join(PERT_BASE, "image", "perturbed_results.csv")

    # Input parameters
    clean_text_params = os.path.join(CLEAN_BASE, "text", "parameters.json")
    clean_image_params = os.path.join(CLEAN_BASE, "image", "parameters.json")

    per_text_params = os.path.join(PERT_BASE, "text", "parameters.json")
    per_image_params = os.path.join(PERT_BASE, "image", "parameters.json")

    scenarios = {
        # Testo perturbato + immagine perturbata
        "both-perturbed": {
            "text_csv": per_text_csv,
            "image_csv": per_image_csv,
            "text_params": per_text_params,
            "image_params": per_image_params,
            "text_state": "perturbed",
            "image_state": "perturbed",
            "subdir": None,
        },

        # Testo clean + immagine perturbata
        "image-perturbed": {
            "text_csv": clean_text_csv,
            "image_csv": per_image_csv,
            "text_params": clean_text_params,
            "image_params": per_image_params,
            "text_state": "clean",
            "image_state": "perturbed",
            "subdir": "image-perturbed",
        },

        # Testo perturbato + immagine clean
        "text-perturbed": {
            "text_csv": per_text_csv,
            "image_csv": clean_image_csv,
            "text_params": per_text_params,
            "image_params": clean_image_params,
            "text_state": "perturbed",
            "image_state": "clean",
            "subdir": "text-perturbed",
        },
    }

    for fusion in ["mean", "min", "max"]:
        fusion_base_dir = os.path.join(PERT_BASE, "late-fusion", fusion)

        for scenario_name, cfg in scenarios.items():
            if cfg["subdir"] is None:
                out_dir = fusion_base_dir
            else:
                out_dir = os.path.join(fusion_base_dir, cfg["subdir"])

            create_late_fusion(cfg["text_csv"], cfg["image_csv"], out_dir, fusion)
            create_late_fusion_parameters(
                text_parameters=cfg["text_params"],
                image_parameters=cfg["image_params"],
                output_dir=out_dir,
                fusion_type=fusion,
                scenario=scenario_name,
                text_state=cfg["text_state"],
                image_state=cfg["image_state"],
                text_csv=cfg["text_csv"],
                image_csv=cfg["image_csv"],
            )