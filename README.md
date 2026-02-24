# RobustnessMultimodal

Official repository for evaluating the **multimodal robustness** of the Themis model through adversarial attacks on both text and image inputs.

---

## Main requirements
- Python == 3.10
- CUDA 11.8 compatible
- torch 2.1.2
- torchvision 0.16.2
- numpy 1.26.4

---

## Usage Guide

### Step 1: Clone the repository
```bash
git clone https://github.com/Davi2082/RobustnessMultimodal.git
cd RobustnessMultimodal
```

---

### Step 2: Create a Python 3.10 environment with pyenv

If you don't already have Python 3.10 installed, you can easily set it up using **pyenv**.

#### Install pyenv (Linux/MacOS)
pyenv should be already available, install Python 3.10 and set it as your active version:
```bash
pyenv install 3.10
pyenv local 3.10
```

#### Install pyenv (Windows)
On Windows, install **pyenv-win** following the official instructions:
```bash
git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\.pyenv
setx PATH "%USERPROFILE%\.pyenv\pyenv-win\bin;%USERPROFILE%\.pyenv\pyenv-win\shims;%PATH%"
```

Then in PowerShell:
```powershell
pyenv install 3.10
pyenv local 3.10
```

After this, verify that the correct Python version is active:
```bash
python --version
# Should print Python 3.10.x
```

---

### Step 3: Create and activate a virtual environment
**On Linux/MacOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

---

### Step 4: Install dependencies

Install the base packages:
```bash
pip install -r requirements.txt
```

Install specific versions of PyTorch and Torchvision. For CUDA 11.8 versions are the following:
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Install NumPy:
```bash
pip install numpy==1.26.4
```

---

### Step 5: Folder and Dataset setup

Before running the code, you need to set up the dataset structure and model weights:

#### Dataset Structure
Create a folder in the `data/` directory for each dataset you want to test. Each dataset folder must contain:
- `images/` → subdirectory containing all dataset images
- `test.*` → test annotations file (can be `.csv`, `.tsv`, or other formats)
- Additional annotation files as needed (e.g., `train.*`, `val.*`)

**Example structure:**
```
Data/
├── Fakeddit/
│   ├── images/
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   ├── test.tsv
│   ├── train.tsv
│   └── val.tsv
├── Recovery/
│   ├── images/
│   ├── test.csv
│   └── ...
└── YourNewDataset/
    ├── images/
    ├── test.csv
    └── ...
```

#### Dataset Class Definition
For each new dataset added to `data/`, you must define the corresponding class and load function in `my_datasets.py`:

1. Create a class named `{DatasetName}_Dataset` (e.g., `YourNewDataset_Dataset`)
2. Create a function named `{datasetname}_load_annotations_file()` (e.g., `yournewdataset_load_annotations_file()`)

#### Model Weights
You must place your pretrained model weights in the `model/` directory

---

### Step 7: Running the attack on Themis

Once the folders are created and all dependencies are installed, you can run the multimodal robustness evaluation with:

```bash
python eval.py
```

By default, this will execute the full pipeline with predefined parameters and save the results to the `results/` directory.

**Optional arguments**
You can customize the execution by passing arguments directly from the command line:
```bash
python eval.py \\
  --name_llm \"phi3:instruct\" \\
  --name_img_embed \"openclip-ViT-B-16\" \\
  --batch_size 8 \\
  --model_path \"model/themis_weights.pt\" \\
  --n_tokens 128 \\
  --pgd_iters 30 \\
  --epsilon 0.0078 \\
  --alpha_factor 2.0 \\
  --dataset_path \"Data/ReCOVery/test.csv\" \\
  --images_path \"Data/ReCOVery/images\" \\
  --results_path \"results/\"
```

**Available parameters**

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--name_llm` | str | Language model used for text corruption (default: Phi-3:instruct) |
| `--name_img_embed` | str | Image encoder name (default: OpenCLIP ViT-B/16 variant) |
| `--batch_size` | int | Batch size for evaluation |
| `--model_path` | str | Path to the model weights |
| `--n_tokens` | int | Maximum token length for text encoder |
| `--merge_tokens` | int | Token merging factor (optional, usually left at 0) |
| `--use_lora` | bool | Enable LoRA fine-tuning (optional) |
| `--lora_alpha`, `--lora_r`, `--lora_dropout` | various | LoRA configuration options — see Themis documentation for details |
| `--set_params` | bool | Whether to use default model parameters (default: True) |
| `--pgd_iters` | int | Number of PGD iterations (default: 30) |
| `--epsilon` | float | Maximum perturbation magnitude (default: 2/255) |
| `--alpha_factor` | float | Step size scaling factor for PGD |
| `--dataset_path` | str | Path to the dataset CSV file |
| `--images_path` | str | Path to the image folder |
| `--results_path` | str | Output directory for evaluation results |

**Example run**
```bash
python eval.py --pgd_iters 30 --epsilon 0.0078 --alpha_factor 2.0 --batch_size 8
```
All results (confusion matrices, metrics, and JSON logs) will be automatically saved under the `results/` directory.

---

## Experiments
The main results of the experiments described in the thesis are available in the [`experiments/`] folder.
Each subfolder includes a `README.md` file that documents the parameters, metrics, and observations related to each run.

---

## Dataset and Model
**Dataset:** ReCOVery, adapted following Is-It-Fake-Or-Not https://github.com/demon-prin/Is-It-Fake-Or-Not.

**Model:** Themis (OpenCLIP ViT-B/16 variant). Model weights must be trained or requested separately.

## Original repository
https://github.com/Davi2082/RobustnessMultimodal

## License
Distributed under the MIT License.
