# PROJECT_DIR=${HOME}"/lang-agnostic/pretraining"
# T5X_DIR=${HOME}"/t5x"
# MODEL_DIR="gs://monobyte/models/english-span3/"
# export PYTHONPATH=${PROJECT_DIR}

# python3 ${T5X_DIR}/t5x/train.py \
#   --gin_search_paths=${PROJECT_DIR} \
#   --gin_file="monobyte_small_pretrain.gin" \
#   --gin.MODEL_DIR=\"${MODEL_DIR}\" \
#   --gin.MIXTURE_OR_TASK_NAME="\"monobyte.pretrain.en\"" \
#   --gin.partitioning.PjitPartitioner.num_partitions=2

PROJECT_DIR=${HOME}"/lang-agnostic/pretraining"
T5X_DIR=${HOME}"/t5x"  # directory where the t5x is cloned.
MODEL_DIR="gs://monobyte/models/english-span20-1M/"
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="monobyte_small_pretrain_moresteps.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\"

