# Getting Started

1. Download [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
2. Open Miniconda and create a Python environment:
```bash
conda create --name hcpdiff python=3.10 -y
conda activate hcpdiff
```
3. Install [pytorch](https://pytorch.org/)
4. Download and install HCP-Diffusion:

::::{tab-set}
:::{tab-item} Install form source
```bash
git clone https://github.com/IrisRainbowNeko/HCP-Diffusion.git
cd HCP-Diffusion
pip install -e .
hcpinit
```
:::
:::{tab-item} Install from pip
```bash
pip install hcpdiff
hcpinit
```
:::
::::