import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requires=[]
with open('requirements.txt') as f:
    for x in f.readlines():
        requires.append(f'"{x.strip()}"')

setuptools.setup(
    name="hcpdiff", # Replace with your own username
    version="1.0.0",
    author="Ziyi Dong",
    author_email="dzy7eu7d7@gmail.com",
    description="A universal Stable-Diffusion toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/7eu7d7/HCP-Diffusion",
    packages=setuptools.find_packages(),
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',

    entry_points={
        'console_scripts': [
            'hcpinit = hcpdiff.tools.init_proj:main'
        ]
    },

    data_files=[
        ('prompt_tuning_template', ['*.txt']),
        ('cfgs', ['*']),
    ],

    install_requires=requires
)