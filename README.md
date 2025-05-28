# Context Robust Knowledge Editing for Language Models

This repository provides an implementation of Context Robust Knowledge Editing (CoRE) on auto-regressive transformers and the Contextual Hop Editing Dataset (CHED) for evaluating context robustness of knowledge editing methods.  
Our work was accepted to ACL 2025 Findings.  
Feel free to open an issue if you find any problems; we are actively developing this repository and will monitor tickets closely.

## Table of Contents
1. [Requirements](#requirements)
2. [Context Robust Knowledge Editing (CoRE)](#context-robust-knowledge-editing-core)
3. [CHED Dataset](#ched-dataset)
4. [Editing and Evaluation](#editing-and-evaluation)
5. [Experimental Results](#experimental-results)
6. [How to Cite](#how-to-cite)

## Requirements

### 🔧 Pip Installation

> **Note:** Please use Python 3.9+ for CoRE.

```shell
git clone https://github.com/holi-lab/CoRE.git
conda create -n CoRE python=3.9.7
conda activate CoRE
pip install -r requirements.txt
````

## Context Robust Knowledge Editing (CoRE)

### Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/8c565e21-90cd-4b06-87f2-3d605b56994b" alt="CoRE Overview" width="400" />
</p>

### Hyperparameters

```yaml
alg_name: CORE
attn_module_tmp: model.layers.{}.self_attn
clamp_norm_factor: 3
device: 0
fact_token: subject_last
kl_factor: 0.0625
layer_module_tmp: model.layers.{}
layer_selection: all
layers:
  - 3
lm_head_module: lm_head
ln_f_module: model.norm
mlp_module_tmp: model.layers.{}.mlp
model_name: meta-llama/Meta-Llama-3-8B-Instruct
model_parallel: true
mom2_adjustment: true
mom2_dataset: wikipedia
mom2_dtype: float32
mom2_n_samples: 100000
mom2_update_weight: 15000
rewrite_module_tmp: model.layers.{}.mlp.down_proj
stats_dir: /data1/home/dellaanima/EasyEdit/stats
v_loss_layer: 31
v_lr: 0.5
v_num_grad_steps: 25
v_weight_decay: 0.001
batch_size: 3

# CORE-specific additional hyperparameters introduced by the CoRE methodology
reg_lambda: 0.04
context: all
ctx_num: 15
layer_range: 28
```

## CHED Dataset

### Dataset Description

<p align="center">
  <img src="https://github.com/user-attachments/assets/81a28698-0460-4d30-acd4-6bca227bf7a2" alt="CHED Dataset" width="400" />
</p>

The Contextual Hop Editing Dataset (CHED) is designed to evaluate the context robustness of knowledge editing methods. The dataset is included in `data/`.

### Dataset Statistics

* **21,782** fact triplets
* **314,385** hop-word prefix context sentences
* **326,730** fact prefix context sentences

### Data Format

The dataset is saved as a list of dictionaries; each dictionary represents one data instance. An example entry from `CHED.json`:

```json
{
  "case_id": "6",
  "counterfact_id": "6",
  "prompt": "{}, that was created in",
  "subject": "Anaal Nathrakh",
  "fact_knowledge": "Birmingham",
  "edited_knowledge": "Philadelphia",
  "relation_id": "P740",
  "rephrased_prompt": "In Wardha he came in close contact with Mahatma Gandhi. Anaal Nathrakh was founded in",
  "locality_prompt": "City of Birmingham Symphony Orchestra, that was created in",
  "locality_ground_truth": "Birmingham",
  "sbj_hop_word": [
    "Back on Black Records",
    "black metal",
    "Season of Mist",
    "Candlelight Records",
    "United Kingdom"
  ],
  "obj_old_hop_word": [
    "Yvonne Mosquito",
    "River Tame",
    "Changchun",
    "GBBHM",
    "West Midlands"
  ],
  "obj_new_hop_word": [
    "Darby",
    "Jim Kenney",
    "Riverton",
    "USPHL",
    "Lower Moreland Township"
  ],
  "sbj_hop_sentence": [
    "The label was founded to support underground artists, Back on Black Records.",
    "This genre is characterized by its intense sound and themes, black metal.",
    "The label expanded its roster significantly over the years, Season of Mist.",
    "Artists under this label have gained international recognition, Candlelight Records.",
    "The music scene in that area has a distinct identity, United Kingdom."
  ],
  "obj_old_hop_sentence": [
    "Yvonne Mosquito first appeared in various documentaries discussing tropical diseases.",
    "Residents often enjoy the beauty of the River Tame throughout the year.",
    "Changchun is famous for its advanced automotive industry in Asia.",
    "The recent events highlighted the importance of GBBHM initiatives for urban development.",
    "Numerous attractions can be found in the West Midlands region."
  ],
  "obj_new_hop_sentence": [
    "The quaint town of Darby is known for its friendly community.",
    "Under Mayor Jim Kenney, the city has seen significant changes.",
    "Located near the river, Riverton offers beautiful waterfront views.",
    "The USPHL provides a platform for aspiring hockey players to showcase their talent.",
    "Lower Moreland Township features several parks and recreational facilities."
  ]
}
```

## Editing and Evaluation

### Quick Start

To apply the CORE editing method to a batch of 1,000 knowledge instances, run:

```bash
python edit_eval/edit_eval.py \
  --editing_method CORE \
  --hparams_dir /data1/home/dellaanima/CoRE/EasyEdit/hparams/CORE/llama3-8b-Instruct.yaml \
  --edit_data_dir /data1/home/dellaanima/CoRE/data/CHED.json \
  --ds_size 1000 \
  --start_sample 1 \
  --end_sample 1000 \
  --save_dir /data1/home/dellaanima/CoRE/edit_eval/output \
  --cuda_device 1 \
  --eval_max_length 50 \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct
```

This will load the specified hyperparameters, process samples 1–1000 from the CHED dataset, perform the edit, and evaluate the results.

### Evaluation Metrics

When the run finishes, you’ll find a JSON file in your `save_dir` with overall and per-category editing statistics. Example:

```json
{
  "AVG_editing_success": 0.8225624310618719,
  "total_editing_time": 10805.295027017593,
  "total_samples": 1000,
  "AVG_SRS_REWRITE": 15.312,
  "REWRITE/editing_success": 0.905,
  "REPHRASE/editing_success": 0.776,
  "SBJ_HOP/editing_success": 0.8832262353201862,
  "OBJ_OLD_HOP/editing_success": 0.7972294719935756,
  "OBJ_NEW_HOP/editing_success": 0.9030437411812134,
  "SBJ/editing_success": 0.874,
  "OBJ_OLD/editing_success": 0.5152,
  "OBJ_NEW/editing_success": 0.9268
}
```

## Experimental Results

TBD

## How to Cite

TBD

```
```
