# 开始使用

1. 下载[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
2. 打开Miniconda并创建python环境
```bash
conda create --name hcpdiff python=3.10 -y
conda activate hcpdiff
```

3. 安装 [pytorch](https://pytorch.org/)

4. 下载并安装HCP-Diffusion

::::{tab-set}
:::{tab-item} 从源码安装
```bash
git clone https://github.com/IrisRainbowNeko/HCP-Diffusion.git
cd HCP-Diffusion
pip install -e .
hcpinit
```
:::
:::{tab-item} 从pip安装
```bash
pip install hcpdiff
# 初始化配置文件
hcpinit
```
:::
::::