import sys
import shutil
import os

def main():
    try:
        shutil.copytree(os.path.join(sys.prefix, 'hcpdiff/cfgs'), r'./cfgs')
        shutil.copytree(os.path.join(sys.prefix, 'hcpdiff/prompt_tuning_template'), r'./prompt_tuning_template')
    except:
        shutil.copytree(os.path.join(sys.prefix, '../hcpdiff/cfgs'), r'./cfgs')
        shutil.copytree(os.path.join(sys.prefix, '../hcpdiff/prompt_tuning_template'), r'./prompt_tuning_template')