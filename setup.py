import setuptools

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

requires = []
with open('requirements.txt', encoding='utf8') as f:
    for x in f.readlines():
        requires.append(f'{x.strip()}')


setuptools.setup(
    name="hcpdiff",
    py_modules=["hcpdiff"],
    version="2.4",
    author="Ziyi Dong",
    author_email="rainbow-neko@outlook.com",
    description="A universal Diffusion toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IrisRainbowNeko/HCP-Diffusion",
    packages=setuptools.find_packages(),
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',

    entry_points={
        'console_scripts': [
            'hcpinit = hcpdiff.tools.init_proj:main',
            'hcp_train = hcpdiff.trainer_ac:hcp_train',
            'hcp_train_1gpu = hcpdiff.trainer_ac_single:hcp_train',
            'hcp_train_ds = hcpdiff.trainer_deepspeed:hcp_train',
            'hcp_run = rainbowneko.infer.infer_workflow:run_workflow',
        ]
    },

    include_package_data=True,

    install_requires=requires
)
