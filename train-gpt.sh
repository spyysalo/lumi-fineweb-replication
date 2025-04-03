#!/bin/bash
#SBATCH --job-name=train-gpt
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=8
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000353
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err

# This is a slurm script for training generative models on LUMI using
# Megatron-LM pretrain_gpt.py. This script defines defaults for
# training a FineWeb-like model (approx. 1.7B parameters) and is
# intended to be reasonably easily modified for other model sizes
# by editing the variables defined in the "MODEL AND PRETRAINING
# CONFIGURATION" section below.
#
# Note that while the script defines default arguments for sbatch
# in the #SBATCH comments above, you can override any of these on the
# command line. For example, to run on 16 nodes:
#
#    sbatch --nodes 16 ./train.sh [...]

######################################################################
#
# ENVIRONMENT SETUP AND GENERAL CONFIGURATION
#
# This section of the script sets up the execution environment (logs,
# container, etc.) and configuration that is independent of the model
# or pretraining setup. It should generally not be necessary to edit
# this section, and you may wish to double-check that you understand
# what you are doing before you do.
#
######################################################################

# If this script is run without sbatch, invoke with sbatch here. This
# also gives us an opportunity to make sure logs/ exists. (If the
# directory where --output and/or --error are directed to doesn't
# exist, the run will fail silently.)
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0" "$@"
    exit
fi

# Bash "strict mode"
# (see http://redsymbol.net/articles/unofficial-bash-strict-mode/)
set -euo pipefail

# When slurm reschedules a job that ended on node failure, it will run
# with the same job ID, clobbering the original logs. Rename the logs
# and include timestamp to avoid this.
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
logfile_basename="${SLURM_JOB_NAME}-${SLURM_JOBID}-${timestamp}"
mv -f "logs/${SLURM_JOBID}.out" "logs/${logfile_basename}.out"
mv -f "logs/${SLURM_JOBID}.err" "logs/${logfile_basename}.err"

# Check if this is a restared run and if so, print the failure
# events/reasons for failed nodes. (This relies on "logs/latest.err"
# pointing to the error log of the failed run.)
if [[ -v SLURM_RESTART_COUNT ]]; then
    failed_node=$(grep 'Node failure' logs/latest.err | awk '{print $NF}')
    if [[ -z ${failed_node:+x} ]]; then
        echo "RUN RESTARTED but no node failure logged"
    else
        failed_node="${failed_node//$'\n'/ }"
        echo "RUN RESTARTED AFTER FAILURE OF NODE(s) $failed_node. Reason:"
        sacctmgr show event where node="$failed_node" format="NodeName,TimeStart,TimeEnd,State,Reason%100"
    fi
fi

# Symlink logs/latest.out and logs/latest.err for convenience and to
# support the above check.
ln -sf "${logfile_basename}.out" "logs/latest.out"
ln -sf "${logfile_basename}.err" "logs/latest.err"

# No modules are needed with the container we are using.
module purge

# Dedicated LUMI container with most of what we need for LLM training
CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif

# Directories to map into container
BIND_DIRS="/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd"

# Avoid conflicts with $HOME/.local
export PYTHONUSERBASE=""

# Compilers in the container
export CC=gcc-12
export CXX=g++-12

# Mask to bind tasks to CPUs for one thread per core
c="fe"
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# PATHS
BASE_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="$BASE_DIR/output"
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints"
TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard/$SLURM_JOB_NAME-$SLURM_JOBID"

mkdir -p "$CHECKPOINT_PATH"    # This needs to exist

# Script that is used to launch on GPU nodes
LAUNCH_SCRIPT="$BASE_DIR/launch.sh"

# Needed for sequence paralellism
# (see https://github.com/NVIDIA/Megatron-LM/issues/533)
export CUDA_DEVICE_MAX_CONNECTIONS=1

# DISTRIBUTED ARGS
# These are used by torch.distributed to allow the different processes
# to find each other. Note that RANK and LOCAL_RANK are also expected,
# but can only be set in the launcher script as the values are
# specific to the process.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999    # TODO add in job ID
export WORLD_SIZE=$SLURM_NTASKS    # Note: only valid if ntasks==ngpus

# OMP THREADING
export OMP_NUM_THREADS=2    # OMP_NUM_THREADS=1 is the safe option
export HSA_ENABLE_SDMA=0

export NCCL_NCHANNELS_PER_PEER=32

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
#export NCCL_DMABUF_ENABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

# DEBUGGING, INCREASE VERBOSITY IN LOGS
# export MIOPEN_ENABLE_LOGGING=1
export PYTHONWARNINGS=ignore
# export TORCH_SHOW_CPP_STACKTRACES=1
# export NCCL_DEBUG=INFO
# export RCCL_KERNEL_COLL_TRACE_ENABLE=1
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG_FILE=$OUTPUT_DIR/nccl-debug-${SLURM_JOB_NAME}-${SLURM_JOBID}.log #Move verbose nccl logging to its own file
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

######################################################################
#
# MODEL AND PRETRAINING CONFIGURATION
#
# This section sets variables that define the model and pretraining
# configuration. These mostly correspond to command-line arguments to
# Megatron-LM/pretrain_gpt.py, and when they do, the names should
# match (e.g. the variable $GLOBAL_BATCH_SIZE gets passed as
# --global-batch-size). This script is intended to be configurable by
# redefining these variables.
#
######################################################################

# DATA
DATA_ROOT="$BASE_DIR/megatron-data"
DATA_PATH="1.0 ${DATA_ROOT}/fineweb-10BT_text_document"
DATA_CACHE_PATH="$DATA_ROOT/cache"
TOKENIZER_MODEL="gpt2-tokenizer"

# MODEL
NUM_LAYERS=24
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=$((4*HIDDEN_SIZE))
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=32    # No GQA when NUM_QUERY_GROUPS=NUM_ATTENTION_HEADS
TIE_WORD_EMBEDDINGS=1
INIT_METHOD_STD=0.02
SEQ_LENGTH=2048
ROTARY_BASE=10000    # Default, recommend larger for higher seq len

# PARALLELISM
PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=1
PROFILE=0

# OPTIMIZER
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_ITERS=500
CLIP_GRAD=1.0
WEIGHT_DECAY=1e-1

# TRAINING
FSDP=0
GLOBAL_BATCH_SIZE=1024
MICRO_BATCH_SIZE=4
RECOMPUTATION=0
TRAIN_TOKENS=350_000_000_000    # TRAIN_SAMPLES computed from this

# SAVING AND EVALUATION
LOG_INTERVAL=1
SAVE_INTERVAL=500
EVAL_INTERVAL=5000
EVAL_ITERS=100

######################################################################
#
# DERIVED CONFIGURATION SETTINGS
#
# The following settings are derived from the configuration above.
# Do set these directly, as they will be overwritten here.
#
######################################################################

# Check that variables are not set (sanity)
confirm_unset() {
    local varname="$1"
    if [ -n "${!varname+x}" ]; then
	echo "Error: variable '$varname' should not be set." >&2
	exit 1
    fi
}
confirm_unset "TRAIN_SAMPLES"
confirm_unset "LR_WARMUP_SAMPLES"
confirm_unset "LR_DECAY_SAMPLES"

# Calculate TRAIN_SAMPLES from TRAIN_TOKENS
TRAIN_TOKENS=${TRAIN_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TRAIN_TOKENS/SEQ_LENGTH))

# Set LR_WARMUP_SAMPLES and LR_DECAY_SAMPLES and based LR_WARMUP_ITERS
# and TRAIN_SAMPLES
LR_WARMUP_SAMPLES=$((LR_WARMUP_ITERS*GLOBAL_BATCH_SIZE))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES

######################################################################
#
# BUILDING COMMAND-LINE ARGUMENTS
#
# The following builds the command-line arguments for
# Megatron-LM/pretrain_gpt.py based on the variables defined above
# (and optionally in any config given to the script). Note that some
# arguments that are not expected to vary are hard-coded here.
#
######################################################################

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --data-cache-path "$DATA_CACHE_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --make-vocab-size-divisible-by 128
    --dataloader-type single
    --num-workers 2   # Some issues with this, lower values are safer
)

MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
)

if [ "$NUM_QUERY_GROUPS" != "$NUM_ATTENTION_HEADS" ]; then
    MODEL_ARGS+=(
        --group-query-attention
        --num-query-groups $NUM_QUERY_GROUPS
    )
fi

if [ "$TIE_WORD_EMBEDDINGS" = "0" ]; then
    MODEL_ARGS+=(
	--untie-embeddings-and-output-weights
    )
fi

if [ "$FSDP" = "1" ]; then
    PARALLEL_ARGS=(
	--use-torch-fsdp2
    )
else
    PARALLEL_ARGS=(
	--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
	--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
	--context-parallel-size $CONTEXT_PARALLEL_SIZE
	--sequence-parallel
	--use-distributed-optimizer
    )
fi

if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS=(
	--use-pytorch-profiler
	--profile-ranks 0
	--profile-step-start 5
	--profile-step-end 7
    )
else
    PROFILE_ARGS=()
fi

MODEL_ARGS+=(
    --use-flash-attn
    --attention-softmax-in-fp32
    --max-position-embeddings $SEQ_LENGTH
    --seq-length $SEQ_LENGTH
    --position-embedding-type rope
    --rotary-base $ROTARY_BASE
    --disable-bias-linear
    --init-method-std $INIT_METHOD_STD
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples $TRAIN_SAMPLES
    --bf16
    --swiglu
    --no-async-tensor-model-parallel-allreduce
    --no-masked-softmax-fusion
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-rope-fusion    # buggy on AMD, do not enable without validating
    --distributed-timeout-minutes 30
    --overlap-grad-reduce
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --adam-beta1 $ADAM_BETA1
    --adam-beta2 $ADAM_BETA2
    --adam-eps $ADAM_EPS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style cosine
    --lr-decay-samples $LR_DECAY_SAMPLES
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --clip-grad $CLIP_GRAD
    --weight-decay $WEIGHT_DECAY
)

OUTPUT_ARGS=(
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
    --tensorboard-dir "$TENSORBOARD_DIR"
    --tensorboard-queue-size 5
    --log-throughput
    --log-progress
    --log-interval $LOG_INTERVAL
)

# Interleaved pipeline scheduling is only possible with pipeline
# parallel degree > 1.
if [ $PIPELINE_MODEL_PARALLEL_SIZE -gt 1 ] && [ $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE -gt 1 ]; then
    PARALLEL_ARGS+=(
	--num-layers-per-virtual-pipeline-stage $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE
    )
fi

if [ "$RECOMPUTATION" = "1" ]; then
    MODEL_ARGS+=(
	--recompute-activations
	--recompute-granularity selective
    )
fi

CHECKPOINT_ARGS=(
    --ckpt-format torch    # "legacy" checkpoints; torch_dist is crashing
#     --async-save    # requires --ckpt-format torch_dist
    --load "$CHECKPOINT_PATH"
    --save "$CHECKPOINT_PATH"
    --save-interval $SAVE_INTERVAL
)

COMMAND=" \
    Megatron-LM/pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
"

######################################################################
#
# Run the command through the launch script with srun.
# Note that any node-specific setup needs to go into the launch script.
#
######################################################################

echo '============= COMMAND: ============='
echo "$COMMAND"
echo '===================================='

echo "START $SLURM_JOBID: $(date)"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

srun \
    --label \
    --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B "$BASE_DIR" \
    -B "$BIND_DIRS" \
    "$CONTAINER" \
    "$LAUNCH_SCRIPT" \
    $COMMAND

echo "END $SLURM_JOBID: $(date)"
