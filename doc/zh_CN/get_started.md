# 开始使用

1. 下载[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
2. 打开Miniconda并创建python环境
```bash
conda create --name hcpdiff python=3.10 -y
conda activate hcpdiff
```
3. 安装pytorch
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. 下载并安装HCP-Diffusion
```bash
git clone https://github.com/7eu7d7/HCP-Diffusion.git
cd HCP-Diffusion
pip install -e .
```