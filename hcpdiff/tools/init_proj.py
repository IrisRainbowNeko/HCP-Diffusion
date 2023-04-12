try:
    # new stdlib in Python3.9
    from importlib.resources import files
except ImportError:
    # third-party package, backport for Python3.9-,
    # need to add importlib_resources to requirements
    from importlib_resources import files

import shutil
import os

def main():
    shutil.copytree(files('hcpdiff').joinpath("cfgs"), r'./cfgs')
    shutil.copytree(files('hcpdiff').joinpath("prompt_tuning_template"), r'./prompt_tuning_template')