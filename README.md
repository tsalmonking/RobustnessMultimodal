# RobustnessMultimodal

Official repository for evaluating the **multimodal robustness** of the *Themis* model through adversarial attacks on both text and image inputs.

---

## Main requirements
- Python == 3.10
- CUDA 11.8 compatible
- torch 2.1.2
- torchvision 0.16.2
- numpy 1.26.4
- accelerate (for distributed or multi-GPU execution)

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
pyenv install 3.10.13
pyenv local 3.10.13
```

#### Install pyenv (Windows)
On Windows, install **pyenv-win** following the official instructions:
```bash
git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%\.pyenv
setx PATH "%USERPROFILE%\.pyenv\pyenv-win\bin;%USERPROFILE%\.pyenv\pyenv-win\shims;%PATH%"
```

Then in PowerShell:
```powershell
pyenv install 3.10.13
pyenv local 3.10.13
```

After this, verify that the correct Python version is active:
```bash
python --version
# Should print Python 3.10.13
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

Install specific versions of PyTorch and Torchvision with CUDA 11.8 support:
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Install NumPy:
```bash
pip install numpy==1.26.4
```

---

### Step 5: Folder setup

You must manually create the following folders before running the code:

- `Data/ReCOVery/images/` → place the images to be used here  
- `model/` → place the model weights to be tested here

---

### Step 6: Configure Accelerate

Before using **Accelerate**, run the configuration wizard to set up your environment:

```bash
accelerate config
```

Recommended answers for a common single-node multi-GPU machine:
- Compute environment: `LOCAL_MACHINE`
- Number of machines: `1`
- Machine rank: `0`
- Use DeepSpeed: `no`
- Use FP16 / mixed precision: `yes` (or `no` if unsupported)
- Number of processes: equal to the number of GPUs you want to use

This will create a configuration file that Accelerate will automatically use.

Alternatively, you can create a custom configuration file, for example:

**accelerate_config.yaml**
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 4
machine_rank: 0
mixed_precision: fp16
use_cpu: false
```

You can then launch the program using:
```bash
accelerate launch --config_file accelerate_config.yaml evaluate_robustness.py
```

---

### Step 7: Running the attack on Themis

Once the folders are created and all dependencies are installed, you can run the multimodal robustness evaluation with:

```bash
python evaluate_robustness.py
```

If you want to use **Accelerate** for multi-GPU or distributed execution, use:

```bash
accelerate launch evaluate_robustness.py
```

The evaluation results will be saved in the **`results/`** directory.

---

## Original repository
[https://github.com/Davi2082/RobustnessMultimodal](https://github.com/Davi2082/RobustnessMultimodal)

---

## License
This project is distributed under the MIT License.
