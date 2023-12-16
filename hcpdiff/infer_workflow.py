import argparse

import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from easydict import EasyDict

from hcpdiff.utils.utils import load_config_with_cli
from .workflow import MemoryMixin
from copy import deepcopy

class WorkflowRunner:
    def __init__(self, cfgs):
        self.cfgs_raw = deepcopy(cfgs)
        self.cfgs = OmegaConf.structured(cfgs, flags={"allow_objects": True})
        OmegaConf.resolve(self.cfgs)
        self.memory = EasyDict(**hydra.utils.instantiate(self.cfgs.memory))
        self.attach_memory(self.cfgs)

    def start(self):
        prepare_actions = hydra.utils.instantiate(self.cfgs.prepare)
        states = self.run(prepare_actions, {'cfgs': self.cfgs_raw})
        actions = hydra.utils.instantiate(self.cfgs.actions)
        states = self.run(actions, states)

    def attach_memory(self, cfgs):
        if OmegaConf.is_dict(cfgs):
            if '_target_' in cfgs and cfgs['_target_'].endswith('.from_memory'):
                cfgs._set_flag('allow_objects', True)
                cfgs['memory'] = self.memory
            else:
                for v in cfgs.values():
                    self.attach_memory(v)
        elif OmegaConf.is_list(cfgs):
            for v in cfgs:
                self.attach_memory(v)

    @torch.inference_mode()
    def run(self, actions, states):
        N_steps = len(actions)
        for step, act in enumerate(actions):
            print(f'[{step+1}/{N_steps}] action: {type(act).__name__}')
            if isinstance(act, MemoryMixin):
                states = act(memory=self.memory, **states)
            else:
                states = act(**states)
            print(f'states: {", ".join(states.keys())}')
        return states

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HCP-Diffusion workflow')
    parser.add_argument('--cfg', type=str, default='')
    args, cfg_args = parser.parse_known_args()
    cfgs = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg

    runner = WorkflowRunner(cfgs)
    runner.start()
