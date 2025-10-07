# RobustnessMultimodal

Official repository for evaluating the **multimodal robustness** of the *Themis* model through adversarial attacks on both text and image inputs.
---

## Main requirements
- Python == 3.10
- CUDA 11.8 compatible
- torch 2.1.2
- torchvision 0.16.2
- numpy 1.26.4

---

## Usage Guide

### Step 1️: Clone the repository
```bash
git clone https://github.com/Davi2082/RobustnessMultimodal.git
cd RobustnessMultimodal
```

### Step 2️: Create and activate a virtual environment
**On Linux/MacOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3️: Install dependencies
Install the base packages:
```bash
pip install -r requirements.txt
```

Install the specific versions of PyTorch and Torchvision with CUDA 11.8 support:
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Install the specific version of NumPy:
```bash
pip install numpy==1.26.4
```

---

### Step 4: Folder setup

You must manually create the following folders before running the code:

- `Data/ReCOVery/images/` → place the images to be used here  
- `model/` → place the model weights to be tested here

---

## Step 5: Running the attack on Themis

Once the folders are created and all dependencies are installed, run the multimodal robustness evaluation with:

```bash
python evaluate_robustness.py
```

This script executes the multimodal attack and evaluates the robustness of the Themis model using the provided weights.

---

## Original repository
[https://github.com/Davi2082/RobustnessMultimodal](https://github.com/Davi2082/RobustnessMultimodal)

---

## License
This project is distributed under the MIT License.
