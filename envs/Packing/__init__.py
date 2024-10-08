# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: Packing-RL
File Name: __init__.py.py
Author: XEH1SGH
Create Date: 4/14/2022
-------------------------------------------------
"""
# from gym.envs.registration import register
# from packing_env import PackingGame
#
# register(
#     id='Pack-v0',
#     entry_point='problems.OnlinePacking:PackingGame',
# )
from .env import PackingEnv

__version__ = "0.0.1"

