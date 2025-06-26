# Context Robust Knowledge Editing for Language Models

This repository provides an implementation of Context Robust Knowledge Editing (CoRE) on auto-regressive transformers and the Contextual Hop Editing Dataset (CHED) for evaluating context robustness of knowledge editing methods. Our work was accepted to **ACL 2025 Findings**.  

## 🔔 News
🌟2025-06-05, the EasyEdit open-source framework (https://github.com/zjunlp/EasyEdit) has added our method CORE. You can easily use CORE through EasyEdit as well.

## Table of Contents
1. [Requirements](#requirements)
2. [Context Robust Knowledge Editing (CoRE)](#context-robust-knowledge-editing-core)
3. [CHED Dataset](#ched-dataset)
4. [Editing and Evaluation](#editing-and-evaluation)
5. [Experimental Results](#-experimental-results)
6. [How to Cite](#how-to-cite)

## Requirements

### 🔧 Pip Installation

> **Note:** Please use Python 3.9+ for CoRE. We have integrated the EasyEdit open-source framework (https://github.com/zjunlp/EasyEdit) so that our CORE method—as well as other baseline editing approaches—runs directly within EasyEdit. As a result, the `requirements.txt` file includes all of EasyEdit’s original dependencies, plus one additional package (`vllm`) to enable faster inference in our `edit_eval` pipeline.



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
Example YAML files are located at `/CoRE/EasyEdit/hparams/CORE/`.

```yaml
alg_name: CORE
attn_module_tmp: model.layers.{}.self_attn
clamp_norm_factor: 3
device: 0
fact_token: subject_last
kl_factor: 0.0625
layer_module_tmp: model.layers.{}
layer_selection: all
layers: [4, 5, 6, 7, 8]  # 3 
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
batch_size : 2000



# CORE 
reg_lambda : 0.01 #0.04
context: all 
ctx_len: 10
ctx_num: 15
ctx_top_k: 5
layer_range: 28

```
> **Note:** When editing more layers, the `reg_lambda` value should be correspondingly smaller.


## CHED Dataset

### 📂 Dataset Preparation
Please unzip the CHED dataset located at ./CoRE/data/CHED.zip.

### Dataset Description

<p align="center">
  <img src="https://github.com/user-attachments/assets/81a28698-0460-4d30-acd4-6bca227bf7a2" alt="CHED Dataset" width="400" />
</p>

The Contextual Hop Editing Dataset (CHED) is designed to evaluate the context robustness of knowledge editing methods.
### Dataset Statistics

* **21,782** fact triplets
* **314,385** hop-word prefix context sentences
* **326,730** fact prefix context sentences

### Data Format

The dataset is saved as a list of dictionaries; each dictionary represents one data instance. An example entry from `CHED.json`:

```json
  {
    "case_id": "13",
    "counterfact_id": "13",
    "prompt": "{}, developed by",
    "subject": "Ferrari F40",
    "fact_knowledge": "Ferrari",
    "edited_knowledge": "Microsoft",
    "relation_id": "P176",
    "rephrased_prompt": "Cartwright attended the Philadelphia School of Design for Women. Ferrari F40 is created by",
    "locality_prompt": "Ferrari 250 GTO is a product of",
    "locality_ground_truth": "Ferrari",
    "sbj_hop_word": [
      "supercar",
      "Pietro Camardella",
      "Leonardo Fioravanti",
      "automobile model",
      "gasoline engine"
    ],
    "obj_old_hop_word": [
      "sports car",
      "supercar",
      "Maranello",
      "racecar constructor",
      "John Elkann"
    ],
    "obj_new_hop_word": [
      "Yammer",
      "Cambria",
      "Office Online",
      "Paul Allen",
      "Bing Maps"
    ],
    "sbj_hop_sentence": [
      "Many enthusiasts eagerly await the release of a new supercar.",
      "Pietro Camardella has been recognized for his innovative designs.",
      "The artwork of Leonardo Fioravanti inspires many in the automotive world.",
      "This is a remarkable example of a cutting-edge automobile model.",
      "Engineers are working to improve the efficiency of a gasoline engine."
    ],
    "obj_old_hop_sentence": [
      "The new model excels in performance and design, appealing to every sports car enthusiast.",
      "High speeds and innovative technology define the modern supercar experience.",
      "Many legendary vehicles have originated from this iconic Italian town, Maranello.",
      "With precision engineering, this company stands out as the leading racecar constructor in the world.",
      "Visionary leadership and commitment to excellence are embodied by John Elkann."
    ],
    "obj_new_hop_sentence": [
      "Yammer is a platform that allows teams to collaborate effectively across different locations.",
      "Cambria is a popular typeface used in various publications and designs.",
      "Office Online provides users with access to essential documents from anywhere.",
      "Paul Allen co-founded one of the most influential technology companies in history.",
      "Bing Maps offers detailed and accurate geographical information for easier navigation."
    ],
    "sbj_sentence": [
      "The iconic design of the Ferrari F40 captivates car enthusiasts around the world.",
      "Performance is a key factor that defines the allure of the Ferrari F40.",
      "Many collectors dream of owning a classic Ferrari F40, renowned for its speed.",
      "Innovation played a crucial role in the engineering of the Ferrari F40.",
      "Enthusiasts often gather at events to celebrate the legacy of the Ferrari F40."
    ],
    "obj_old_sentence": [
      "The iconic vehicle brand, Ferrari, is synonymous with speed and luxury.",
      "Many car enthusiasts dream of owning a Ferrari one day.",
      "Ferrari has a rich history in motorsport, dominating various racing events.",
      "The design of a Ferrari reflects a perfect blend of art and engineering.",
      "A limited edition of Ferrari attracts collectors from all over the globe."
    ],
    "obj_new_sentence": [
      "Microsoft revolutionized personal computing with its innovative software solutions.",
      "Many businesses rely on Microsoft products to enhance productivity and collaboration.",
      "The rapid growth of cloud services has been driven by Microsoft and its competitors.",
      "Developers flock to Microsoft to create applications for its vast ecosystem.",
      "Microsoft continues to dominate the tech industry with new advancements and acquisitions."
    ]
  },
```

## Editing and Evaluation

### Quick Start

To apply the CORE editing method to a batch of 2,000 knowledge instances, run:

```bash
python ./edit_eval/edit_eval.py \
  --editing_method CORE \
  --hparams_dir ./CoRE/EasyEdit/hparams/CORE/llama3-8b-Instruct.yaml \
  --edit_data_dir ./CoRE/data/CHED.json \
  --ds_size 2000 \
  --start_sample 1 \
  --end_sample 2000 \
  --save_dir ./CoRE/edit_eval/output \
  --cuda_device 0 \
  --eval_max_length 50 \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct
```
> **Note:** This evaluation departs from traditional *probability-based* methods that have commonly been used in prior knowledge editing paper. Instead, it adopts a more **strict and realistic generation-based** evaluation strategy, better reflecting real-world usage. Specifically, an edit is considered successful if the model, within `--eval_max_length` tokens, generates a response that contains the edited knowledge but excludes the original knowledge.


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
  "LOCALITY/editing_success": 0.4,
  "SBJ_HOP/editing_success": 0.8832262353201862,
  "OBJ_OLD_HOP/editing_success": 0.7972294719935756,
  "OBJ_NEW_HOP/editing_success": 0.9030437411812134,
  "SBJ/editing_success": 0.874,
  "OBJ_OLD/editing_success": 0.5152,
  "OBJ_NEW/editing_success": 0.9268
}

```
## Experimental Results 

The following table presents the editing success rates of three methods — **CoRE**, **MEMIT**, and **AlphaEdit** — evaluated on 2,000 batch-edited instances from the **CHED** dataset. **SBJ\_HOP**, **OBJ\_OLD\_HOP**, **OBJ\_NEW\_HOP**, **SBJ**, **OBJ\_OLD**, and **OBJ\_NEW** represent six different types of prefix contexts in CHED.

| **Metric**                        | **CoRE** | **MEMIT** | **AlphaEdit** |
| --------------------------------- | :------: | :-------: | :-----------: |
| Average editing success (%)       | **83.0** |    81.3   |      72.5     |
| REWRITE editing success (%)       | **89.9** |    89.0   |      87.0     |
| REPHRASE editing success (%)      | **81.7** |    79.4   |      66.5     |
| SBJ\_HOP editing success (%)      | **86.1** |    83.7   |      73.5     |
| OBJ\_OLD\_HOP editing success (%) | **77.8** |    74.3   |      63.5     |
| OBJ\_NEW\_HOP editing success (%) | **91.8** |    90.5   |      85.7     |
| SBJ editing success (%)           | **89.0** |    88.4   |      74.2     |
| OBJ\_OLD editing success (%)      | **53.8** |    53.1   |      40.5     |
| OBJ\_NEW editing success (%)      | **93.6** |    91.9   |      89.1     |
> **Note:** Bold values indicate the highest performance for each metric across the three methods.



## How to Cite

If you use CHED or CORE, please cite the following:

```bibtex
@misc{park2025contextrobustknowledgeeditinglanguage,
  title        = {Context‐Robust Knowledge Editing for Language Models},
  author       = {Haewon Park and Gyubin Choi and Minjun Kim and Yohan Jo},
  year         = {2025},
  eprint       = {2505.23026},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2505.23026},
}
