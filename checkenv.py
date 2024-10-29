"""
Copyright (c) 2024 Aleksandar Topuzovic
Email: aleksandar.topuzovic@gmail.com

This software is provided "as is," without any express or implied warranty.
In no event shall the authors be liable for any damages arising from the use
of this software.
"""

from stable_baselines3.common.env_checker import check_env

from tetrisenv import (BaseRewardTetrisEnv, MyTetrisEnv, MyTetrisEnv2,
                       StandardReward2TetrisEnv, StandardRewardTetrisEnv)

if __name__ == "__main__":
    print("Checking the base env ...")
    base_env = BaseRewardTetrisEnv()
    check_env(base_env)

    print("Checking the standard env ...")
    env = StandardRewardTetrisEnv()
    check_env(env)

    print("Checking the standard env2 ...")
    env = StandardReward2TetrisEnv()
    check_env(env)

    print("Checking the my env ...")
    env = MyTetrisEnv()
    check_env(env)

    print("Checking the my env2 ...")
    env = MyTetrisEnv2()
    check_env(env)
