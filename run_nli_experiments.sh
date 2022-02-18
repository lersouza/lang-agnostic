BASE_PATH=${1:-"./configs/"}
CACHE_DIR=${2:-"./.cache"}
DEBUG_MODE=${3:-"0"}

EXPERIMENTS=$(find $BASE_PATH -type f)
NUM_OF_EXPETIMENTS=$(echo "$EXPERIMENTS" | wc -l)
SEEDS=(42 123 1337)

echo "Running all experiments in ${BASE_PATH}. Cache dir is $CACHE_DIR"
echo "====> Found $NUM_OF_EXPETIMENTS Experiments."
echo ""

for exp in $EXPERIMENTS; do
    for seed in "${SEEDS[@]}"; do
        echo "====================================================="
        echo "Starting experiment $exp with seed $seed ..."
        
        if [ $DEBUG_MODE == "0" ]; then
            python src/train_nli.py fit --config $exp --seed_everything $seed --data.init_args.cache_dir=$CACHE_DIR
        else
            echo "Will run: python src/train_nli.py fit --config $exp --seed_everything $seed --data.init_args.cache_dir=$CACHE_DIR"
        fi

        echo "Finished experiment $exp with seed $seed!"
        echo "====================================================="
        echo ""
    done
done

