#! /bin/sh

CONFIG=$1
CONFIG_FILE="configs/$CONFIG.yaml"

python src/train_nli.py fit --config $CONFIG_FILE