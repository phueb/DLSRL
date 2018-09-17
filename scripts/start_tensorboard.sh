#!/usr/bin/env bash

# start tensorboard
ssh s76 "nohup python3 /home/ph/.local/bin/tensorboard --logdir=/home/lab/cluster/tensorboard/dlsrl > /dev/null 2>&1 &"
