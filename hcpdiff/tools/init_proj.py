import sys
import shutil
import os

def main():
    prefix = sys.prefix
    if not os.path.exists(os.path.join(prefix, 'hcpdiff')):
        prefix = os.path.join(prefix, 'local')
    try:
        if os.path.exists(r'./cfgs'):
            shutil.rmtree(r'./cfgs')
        if os.path.exists(r'./prompt_tuning_template'):
            shutil.rmtree(r'./prompt_tuning_template')
        shutil.copytree(os.path.join(prefix, 'hcpdiff/cfgs'), r'./cfgs')
        shutil.copytree(os.path.join(prefix, 'hcpdiff/prompt_tuning_template'), r'./prompt_tuning_template')
    except:
        try:
            shutil.copytree(os.path.join(prefix, '../hcpdiff/cfgs'), r'./cfgs')
            shutil.copytree(os.path.join(prefix, '../hcpdiff/prompt_tuning_template'), r'./prompt_tuning_template')
        except:
            this_file_dir = os.path.dirname(os.path.abspath(__file__))
            shutil.copytree(os.path.join(this_file_dir, '../../cfgs'), r'./cfgs')
            shutil.copytree(os.path.join(this_file_dir, '../../prompt_tuning_template'), r'./prompt_tuning_template')
