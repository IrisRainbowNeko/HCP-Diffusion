import setuptools
import os
import platform

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

requires = []
with open('requirements.txt', encoding='utf8') as f:
    for x in f.readlines():
        requires.append(f'{x.strip()}')

if platform.system().lower() == 'windows':
    requires.append('bitsandbytes-windows')
else:
    requires.append('bitsandbytes')

def get_data_files(data_dir, prefix=''):
    file_dict = {}
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if prefix+root not in file_dict:
                file_dict[prefix+root] = []
            file_dict[prefix+root].append(os.path.join(root, name))
    return [(k, v) for k, v in file_dict.items()]


setuptools.setup(
    name="hcpdiff",
    py_modules=["hcpdiff"],
    version="0.6.0",
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
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',

    entry_points={
        'console_scripts': [
            'hcpinit = hcpdiff.tools.init_proj:main'
        ]
    },

    data_files=[
        *get_data_files('prompt_tuning_template', prefix='hcpdiff/'),
        *get_data_files('cfgs', prefix='hcpdiff/'),
    ],

    install_requires=requires
)
