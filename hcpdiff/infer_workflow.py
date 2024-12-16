import argparse

import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from easydict import EasyDict

from hcpdiff.utils import load_config_with_cli, is_list
from copy import deepcopy
import hcpdiff.workflow

class WorkflowRunner:
    def __init__(self, cfgs):
        self.cfgs_raw = deepcopy(cfgs)
        self.cfgs = OmegaConf.structured(cfgs, flags={"allow_objects": True})
        OmegaConf.resolve(self.cfgs)
        self.memory = EasyDict(**hydra.utils.instantiate(self.cfgs.memory))
        #self.attach_memory(self.cfgs)

    def start(self):
        states = {'cfgs': self.cfgs_raw}
        for action_name in self.cfgs.actions:
            cfg_action = self.resolve_action_ref(self.cfgs, self.cfgs[action_name])
            self.attach_memory(cfg_action)
            actions = hydra.utils.instantiate(cfg_action)
            states = self.run(actions, states)

    def attach_memory(self, cfgs):
        if OmegaConf.is_dict(cfgs):
            if '_target_' in cfgs and cfgs['_target_'].endswith('.from_memory'):
                cfgs._set_flag('allow_objects', True)
                cfgs['memory'] = self.memory
            else:
                for v in cfgs.values():
                    self.attach_memory(v)
        elif is_list(cfgs):
            for v in cfgs:
                self.attach_memory(v)

    def resolve_action_ref(self, cfgs, cfg_action):
        new_cfg_action = []
        for act in cfg_action:
            if isinstance(act, str):
                new_cfg_action.extend(self.resolve_action_ref(cfgs, cfgs[act]))
            elif is_list(act):
                new_cfg_action.append(self.resolve_action_ref(cfgs, act))
            elif 'actions' in act:
                act['actions'] = self.resolve_action_ref(cfgs, act['actions'])
                new_cfg_action.append(act)
            else:
                new_cfg_action.append(act)
        return new_cfg_action


    @torch.inference_mode()
    def run(self, actions, states):
        N_steps = len(actions)
        for step, act in enumerate(actions):
            print(f'[{step+1}/{N_steps}] action: {type(act).__name__}')
            states = act(memory=self.memory, **states)
            #print(f'states: {", ".join(states.keys())}')
        return states

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HCP-Diffusion workflow')
    parser.add_argument('--cfg', type=str, default='')
    args, cfg_args = parser.parse_known_args()
    cfgs = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg

    runner = WorkflowRunner(cfgs)
    runner.start()
