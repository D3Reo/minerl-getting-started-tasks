# Tasks for getting started with MineRL

This repository contains examples and small tasks on getting
started with MineRL environment.

To begin, install the [requirements for MineRL](https://minerl.readthedocs.io/en/latest/tutorials/index.html),
and then install Python requirements with with `pip install -r requirements.txt`. We also have Colab notebooks
in case installing these libraries is not possible.

If you have any questions, you can reach us on [Discord](https://discord.com/invite/BT9uegr).
If you spot typos/bugs in any of the tasks or this repo, do tell us via Github issues!

## Tasks

Stars indicate the difficulty of the task. Click the task to see more details.

**:star: Getting started with MineRL.**
  * Start by playing bit of Minecraft via MineRL with `playing_with_minerl.py` script.
  * Check out `getting_familiar_with_minerl_and_gym.py` and follow the instructions to get familiar with the agent-environment (Gym) API.
  * You can find the latter task on Colab [here](https://colab.research.google.com/drive/11CVCeb7f0P2nqcgWGLG1wDZcE3AxngxL?usp=sharing).

**:star: Improve Intro baseline of the Diamond competition.**
  * Step-by-step instructions on how to improve a simple, fully-scripted agent for obtaining wood and stone in the MineRLObtainDiamond-v0 task.
  * Start out by [opening this document](https://docs.google.com/document/d/12d0jMnsoR5xjyye4Rlpo84yJOZRMbfSYOb17OWOJdFw/edit) and following the instructions.
</details>

**:star::star: Implementing behavioural cloning from (almost) scratch.**
  * Start by opening up `behavioural_cloning.py` and following the instructions at the beginning of the file.
  * You can also find the task on Colab [here](https://colab.research.google.com/drive/1JQ9suwMe-TnyBoDjhdydI6Ic35-m6NLh?usp=sharing).
  * You can find a crude reference answers [in this Colab notebook](https://colab.research.google.com/drive/1JQ9suwMe-TnyBoDjhdydI6Ic35-m6NLh?usp=sharing).
  This task is built on the [BC + scripted baseline solution](https://github.com/KarolisRam/MineRL2021-Intro-baselines/blob/main/standalone/BC_plus_script.py).

**:star::star::star: Learn how to use stable-baselines and imitation libraries with MineRL.**
  * This walk-through demonstrates how to combine well-established reinforcement learning ([stable-baselines3](https://github.com/DLR-RM/stable-baselines3)) and imitation learning ([imitation](https://github.com/HumanCompatibleAI/imitation)) libraries with MineRL to train more sophisticated agents.
  * Start by opening [this Colab link](https://colab.research.google.com/drive/13_jI8YLk9ATRQSd7_3rV5rOsll7jsSz0).

**:star::star::star: Improve Research baseline of the Diamond competition.**
  * Similar to the second task here, but in a more difficult setting where you may not manually encode actions.
  * Get started by opening [this documentation](https://docs.google.com/document/d/1BxKAFZN1-qfc83GjVYMdsJamU01sngn2LlreuvdxWu0/edit?usp=sharing).
