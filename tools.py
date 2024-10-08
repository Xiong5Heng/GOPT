import os
import argparse
import time
import random
from shutil import copyfile, copytree
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
from gymnasium.envs.registration import register


def backup(time_str, args, upper_policy=None):
    if args.mode == "test":
        target_dir = os.path.join('./logs/evaluation', time_str)
    else:
        target_dir = os.path.join('./logs/experiment', time_str)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # TODO
    copyfile('config.py', os.path.join(target_dir, 'config.py'))

    # gym_path = './problems'
    # env_name = args.id.split('-v')
    # env_name = env_name[0] + env_name[1]
    # env_path = os.path.join(gym_path, env_name)
    # copytree(env_path, os.path.join(target_dir, env_name))

    if upper_policy is not None:
        torch.save(upper_policy.state_dict(),
                   os.path.join(args.model_save_path, time_str, 'upper-first-' + time_str + ".pt"))


def registration_envs():
    register(
        id='OnlinePack-v1',
        entry_point='envs.Packing:PackingEnv',
    )
    

def load_policy(load_path, model, device="cpu"):
    print(f"load model from: {load_path}")
    assert os.path.exists(load_path), 'File does not exist'
    pretrained_state_dict = torch.load(load_path, map_location=device)
    if len(pretrained_state_dict) == 2:
        pretrained_state_dict, ob_rms = pretrained_state_dict

    load_dict = {}
    for k, v in pretrained_state_dict.items():
        if 'actor.embedder.layers' in k:
            load_dict[k.replace('module.weight', 'weight')] = v
        else:
            load_dict[k.replace('module.', '')] = v

    load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
    load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}

    model.load_state_dict(load_dict, strict=True)
    print('Loading pre-train upper model', load_path)
    return model


def set_seed(seed: int, cuda: bool = False, cuda_deterministic: bool = False):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if cuda and torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)


class CategoricalMasked(torch.distributions.Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.device = logits.device
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(self.device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e18).to(self.device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(self.device))
        return -p_log_p.sum(-1)


if __name__ == '__main__':
    registration_envs()
    from gymnasium import envs
    envids = [spec.id for spec in envs.registry.all()]
    print(envids)
