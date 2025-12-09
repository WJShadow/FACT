# FACT: Foundation model for calcium-like transient extraction and neuronal footprint segmentation
![FACT_Sub](sub/FACT.jpg)

[![CC BY 4.0][cc-by-shield]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

FACT (Find Any Calcium-like Transients) is a foundation model designed for calcium-like transient extraction from functional imaging videos. This repository contains the implementation of our model, including installation, inference, and evaluation guidance.

---

## 🔧 Installation & Environment Setup

### Prerequisites
- **Python**: 3.8.18 (required)
- **Package Manager**: We recommend using `conda` or `mamba` for environment management

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/FACT.git
cd FACT
```

### Step 2: Create Conda Environment
We provide two methods to set up the environment:

#### Method A: Using our environment file (Recommended)
```bash
conda env create -f Installation/environment_modif.yml
conda activate FACT
```

#### Method B: Manual Setup
If you prefer to create the environment manually:
```bash
conda create -n FACT python=3.8.18
conda activate FACT

# Install core dependencies
conda install pytorch=2.1.0=py3.8_cuda12.1_cudnn8_0 -c pytorch
pip install numpy==1.24.4 
pip install jupyter notebook tqdm

# Install additional packages from our requirements
conda env update -f environment_modif.yml
```

### Step 3: Verify Installation
```python
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import sys; from monai.config import print_config; print_config()"
```

---

## 📁 Project Structure
```
FACT/
├── data/                     # Data used for demo and reproduction of evaluation
├── IO/                       # Read and write data/label 
├── model/                    # Model architectures
├── ModelInference/           # Sliding-window inference code
├── ModelParams/              # Pretrained and Finetunned nework parameters of FACT
├── Preprocessing/            # Simple preprocessing code
├── PostSlice/                # Postprocessing code
├── UI/                       # Visualization code of data and inference results
├── utils/                    # Utility functions
├── Installation/             # Environment setup files
├── LICENSE
├── README.md
└── *Notebook reproduction code*
```

---

## 🚀 Quick Start

### 1. Data Preparation
```python
from IO.Read_tif import load_tiff
from Preprocessing.Normalize import normal
from Preprocessing.Thresh import thresh_max, thresh_min
# (1) Place your datasets in the data/ directory (currently supporting .tif, .tiff, .nii, .nii.gz)
# (2) Call corresponding reader in IO, i.e.
data_path = 'data/STA_Evaluation/Vid01.tiff'  
input_img = load_tiff(data_path)  
# (3) Normalization of input data
#     Normalize the input data to a 0–1 range using the common min-max scaling method. 
#     One simple way is referring to ImageJ's auto-adjust results for reference: Image-Adjust-Brightness/Contrast-[Auto]
#     To avoid introducing artificial noises, we recommend threshing only the maximum
input_img = thresh_max(input_img, max_value)
input_img = thresh_min(input_img, min_value) # Optional
input_img = normal(input_img)

```

### 2. Inference
```python
from ModelInference.SWInf import sliding_window_inference
from model.TS_Net_change import FACT_Net
# (1) Load model and parameters 
model_pth = "ModelParams/FACT_Modelparams.pt"
model = FACT_Net(
    img_size=(128,64,64),
    in_channels=1,
    out_channels=2,
    init_dim=3,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    use_checkpoint=True,
).to(device)

weight = torch.load(model_pth, map_location = 'cpu')
state_dict = weight["state_dict"]
model.load_state_dict(state_dict)

# (2) Model Inference
input_img_tensor = torch.from_numpy(input_img)
model.eval()
with torch.no_grad():
    # test_inputs = torch.unsqueeze(input_img_tensor, 1).cuda(device=device)
    test_inputs = torch.unsqueeze(input_img_tensor, 0)
    test_inputs = torch.unsqueeze(test_inputs, 1)
    test_outputs = sliding_window_inference(
        test_inputs, (128, 64, 64), 32, model, overlap=[0.6,0.2,0.2], progress=True, mode="constant", 
        device=torch.device('cpu'), sw_device=device, 
    )

```

---

## 📊 Evaluation

To reproduce our paper's results:

Please follow the instruction in each jupyter nootbook for inference and evaluation
---

## ⚖️ License & Usage

### Usage Restrictions
This code is released for **academic and personal research purposes only**. The following restrictions apply:

1. **Commercial Use Prohibited**: You may not use this code, model weights, or derivatives for commercial purposes without explicit written permission from the authors.

2. **Redistribution Restrictions**: You may not redistribute this code or model weights without including this license and attribution.

3. **Ethical Use**: Users must ensure their applications do not violate ethical guidelines or cause harm.

### Citation
If you use FACT in your research, please cite our paper:

```bibtex
@article{FACT2025,
  title={},
  author={},
  journal={},
  year={},
  doi={}
}
```

*Note: Citation details will be updated upon paper acceptance.*

---

## 🤝 Contributing

We welcome contributions to improve FACT. Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

---

## 🐛 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size in SW inference configuration (default=32 for VRAM>=24GB)
2. **Missing dependencies**: Ensure all packages in `environment_modif.yml` are installed, for errors during installation please try installing corresponding packages manually
3. **Python version mismatch**: Verify Python version is exactly 3.8.18

### Get Help
- Open an Issue on GitHub for bugs or questions
- Contact the authors for academic collaboration inquiries

---

---
## Acknowledgment
We appreciate contributors of Project MONAI for providing fantastic open-source workflow platform. 

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Monai Core Documentation](https://docs.monai.org.cn/en/stable/)

---

*Last Updated: December 2025*  
*Maintainer: William*