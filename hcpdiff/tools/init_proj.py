from rainbowneko.tools.init_proj import copy_package_data

def main():
    copy_package_data('hcpdiff', 'cfgs', './cfgs')
    copy_package_data('hcpdiff', 'prompt_template', './prompt_template')