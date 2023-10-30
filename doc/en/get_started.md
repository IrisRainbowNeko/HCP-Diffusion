# Getting Started

1. Download [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
2. Open Miniconda and create a Python environment:
```bash
conda create --name hcpdiff python=3.10 -y
conda activate hcpdiff
```
3. Install PyTorch:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
4. Download and install HCP-Diffusion:
```bash
git clone https://github.com/7eu7d7/HCP-Diffusion.git
cd HCP-Diffusion
pip install -e .
```