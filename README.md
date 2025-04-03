# LUMI FineWeb replication

Scripts and instructions for partially replicating the original [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) experiments on LUMI using Megatron-LM.

These instructions assume access to LUMI and a custom container, and are unlikely to be particularly useful otherwise.

## Setup

As these steps involve fairly large datasets, you should probably be working on a project scratch directory. These instructions assume that the project is `project_462000353`, and if you are using a different project, you will need to replace that identifier with your project identifier in examples and slurm scripts for this to work.

### Create and enter working directory in project scratch

```
mkdir -p /scratch/project_462000353/$USER/fineweb-repro
cd /scratch/project_462000353/$USER/fineweb-repro
```

The rest of the instructions assume the above is your working directory unless stated otherwise.

### Load pytorch module

The following module provided by [CSC](https://csc.fi/) includes most of the required libraries, including pytorch and transformers.

```
module use /appl/local/csc/modulefiles
module load pytorch
```

These instructions assume that this module is loaded.

### Clone ROCm Megatron

On LUMI, we'll need the ROCm fork of Megatron.

```
git clone https://github.com/ROCm/Megatron-LM
```

The latest commit when writing these instructions was `99bb7a9`. If something breaks in Megatron, try to check out this specific commit.

## Download FineWeb sample

We'll here use the 10 billion token FineWeb sample as an example to keep download and processing relatively quick. As this is comparatively small, we'll here simply work on a login node and download the data with `load_dataset` and save with `to_json`. If you want to use a larger sample or the entire dataset, you should probably use a compute node and e.g. `datatrove` (see example [here](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2#using-%F0%9F%8F%AD-datatrove)).

Set the HF cache to a subdirectory of the working directory so that the download doesn't exhaust the limited space that's available in your home directory. (If you already have `HF_HOME` set to e.g. some shared cache directory, skip this step.)

```
export HF_HOME=$PWD/cache
```

Download in an interactive Python session (start with `python3`). This should take about 30 minutes.

```
from datasets import load_dataset
d = load_dataset('HuggingFaceFW/fineweb', 'sample-10BT', split='train')
d.to_json('fineweb-10BT.jsonl')
```

Check that the downloaded data has the expected number of lines.

```
wc -l fineweb-10BT.jsonl
```

This should give `14868862`. (`md5sum` was `1086778b352dacb729517ca328b14c62`.)

## Preprocess data

Megatron uses a specialized binary format for input data, and we'll use the script `preprocess_data.py` provided with Megatron to convert the JSONL into this format.

### Start interactive session on CPU node

We'll use several workers to speed up the conversion, so we'll start an interactive session on a compute node to avoid using too much CPU on a login node.

(As of this writing the queues on the `standard` partition were shorter than those on `small`, so this uses `standard`, but `small` may be faster for you.)

```
srun --account=project_462000353 --partition=standard --nodes=1 --cpus-per-task=32 --time=00:30:00 --mem=100G --pty bash
```

### Run preprocessing script

This should be executed on the compute node and take about 30 minutes.

```
python3 Megatron-LM/tools/preprocess_data.py --input fineweb-10BT.jsonl --output fineweb-10BT --tokenizer-type HuggingFaceTokenizer --tokenizer-model gpt2 --append-eod --log-interval 10000 --workers 32
```

After the preprocessing completes, terminate the interactive session on the compute node and return to the login node (`exit` or CTRL-D).

The preprocessing should have created two files, `fineweb-10BT_text_document.bin` and `fineweb-10BT_text_document.idx`. The sizes of these should be as follows (use e.g. `du -h fineweb-10BT_text_document.*`):

```
20G	fineweb-10BT_text_document.bin
284M	fineweb-10BT_text_document.idx
```

Let's move these to a subdirectory to keep things cleaner. (This is also expected by the training script.)

```
mkdir megatron-data
mv fineweb-10BT_text_document.* megatron-data/
```

## Schedule pretraining run

(TODO)
