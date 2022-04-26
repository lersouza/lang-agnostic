#!/bin/bash

BASE_PATH=${1:-"./configs/"}
CACHE_DIR=${2:-"./.cache"}
DEBUG_MODE=${3:-"0"}

EXPERIMENTS=$(find $BASE_PATH -type f -name "*.yaml")
NUM_OF_EXPETIMENTS=$(echo "$EXPERIMENTS" | wc -l)
SEEDS=(42 123 1337)

echo "Running all experiments in ${BASE_PATH}. Cache dir is $CACHE_DIR"
echo "====> Found $NUM_OF_EXPETIMENTS Experiments."
echo ""

echo "Login to Wandb"
wandb login
echo ""

for exp in $EXPERIMENTS; do
    for seed in "${SEEDS[@]}"; do
        echo "====================================================="
        echo "Starting experiment $exp with seed $seed ..."

        if [ -f "$exp.$seed.finished" ]; then
            echo "Experiment $exp already finished previously!"
            echo "====================================================="

            continue
        fi
        
        CURRENT_RUN_ID=$(cat $exp.$seed.inprogress 2>/dev/null)
        COMMAND="python src/train_nli.py fit --config $exp --seed_everything $seed --data.init_args.cache_dir=$CACHE_DIR"
        
        if [ ! -z "$CURRENT_RUN_ID" ]; then
            COMMAND="$COMMAND --trainer.logger=WandbLogger --trainer.logger.id=$CURRENT_RUN_ID"

            if [ -f "./nli/$CURRENT_RUN_ID/checkpoints/last.ckpt" ]; then
                COMMAND="$COMMAND --ckpt_path=./nli/$CURRENT_RUN_ID/checkpoints/last.ckpt"
            fi
        fi

        if [ $DEBUG_MODE == "1" ]; then
            COMMAND="echo $COMMAND"
        fi

        $COMMAND && touch "$exp.$seed.finished"

        echo "Finished experiment $exp with seed $seed!"
        echo "====================================================="
        echo ""
    done
done

