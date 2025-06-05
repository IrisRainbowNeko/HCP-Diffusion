import argparse

from rainbowneko.train.trainer import TrainerSingleCard

from hcpdiff.trainer_ac import HCPTrainer, load_config_with_cli

class HCPTrainerSingleCard(TrainerSingleCard, HCPTrainer):
    pass

def hcp_train():
    import subprocess
    parser = argparse.ArgumentParser(description='HCP-Diffusion Launcher')
    parser.add_argument('--launch_cfg', type=str, default='cfgs/launcher/single.yaml')
    args, train_args = parser.parse_known_args()

    subprocess.run(["accelerate", "launch", '--config_file', args.launch_cfg, "-m",
                    "hcpdiff.trainer_ac_single"] + train_args, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HCP Diffusion Trainer')
    parser.add_argument("--cfg", type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()

    parser, conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = HCPTrainerSingleCard(parser, conf)
    trainer.train()
