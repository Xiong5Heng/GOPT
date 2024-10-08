import os
curr_path = os.path.dirname(os.path.abspath(__file__))

import argparse

from omegaconf import OmegaConf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="cfg/config.yaml")
    parser.add_argument('--ckp', type=str, default=None, 
                        help="Path to the model to be tested")
    parser.add_argument('--no-cuda', action='store_true',
                        help='Cuda will be enabled by default')
    parser.add_argument('--device', type=int, default=0, 
                        help='Which GPU will be called')
    parser.add_argument('--test-episode', type=int, default=1000, 
                        help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment while testing')
    
    args = parser.parse_args()

    try:
        args.config = os.path.join(curr_path, args.config)
        cfg = OmegaConf.load(args.config)
    except FileNotFoundError:
        print("No configuration file found")
    
    box_small = int(max(cfg.env.container_size) / 10)
    box_big = int(max(cfg.env.container_size) / 2)
    # box_range = (5, 5, 5, 25, 25, 25)
    box_range = (box_small, box_small, box_small, box_big, box_big, box_big)

    if cfg.get("env.step") is not None:
        step = cfg.env.step
    else:
        step = box_small

    box_size_set = []
    for i in range(box_range[0], box_range[3] + 1, step):
        for j in range(box_range[1], box_range[4] + 1, step):
            for k in range(box_range[2], box_range[5] + 1, step):
                box_size_set.append((i, j, k))
    
    cfg.env.box_small = box_small
    cfg.env.box_big = box_big
    cfg.env.box_size_set = box_size_set
    cfg.cuda = not args.no_cuda 

    cfg = OmegaConf.merge(cfg, vars(args))

    return cfg


if __name__ == "__main__":
    args = get_args()
    print(args.train.reward_type)