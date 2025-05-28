# Context Robust Knowledge Editing for Language Models
This repository provides an implementation of Context Robust Knowledge Editing (CoRE) on auto-regressive transformers (GPU-only) and Contextual Hop Editing Dataset (CHED) dataset for evaluating context robustness of knowledge editing methods.
Feel free to open an issue if you find any problems; we are actively developing this repository and will monitor tickets closely.

## Table of Contents
1. [Requirements](#requirements)
2. [Context Robust Knowledge Editing (CoRE)](#context-robust-knowledge-editing-core-1)
3. [CHED Dataset](#ched-dataset)
4. [Editing and Evaluation](#editing-and-evaluation)
5. [Experimental Results](#experimental-results)
6. [How to Cite](#how-to-cite)


## Requirements

### 🔧 Pip Installation
**Note: Please use Python 3.9+ for CoRE** To get started, simply install conda and run:

To get started, simply install conda and run:

```shell
git clone https://github.com/holi-lab/CoRE.git
conda create -n CoRE python=3.9.7
conda activate CoRE
pip install -r requirements.txt
```


## Context Robust Knowledge Editing (CoRE)

### Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/8c565e21-90cd-4b06-87f2-3d605b56994b" alt="CoRE Overview" width="400" />
</p>

TBD

### hperparameters
TBD


## CHED Dataset

### Dataset Description
![image](https://github.com/user-attachments/assets/81a28698-0460-4d30-acd4-6bca227bf7a2)
The Contextual Hop Editing Dataset (CHED) is designed to evaluate the context robustness of knowledge editing methods.

The dataset is included in `data/`.

### Dataset Statistics

`CHED` contains 21,782 instances used to evaluate knowledge editing methods in the presence of preceding context, including:
- 21,782 fact triplets
- 314,385 hop-word prefix context sentences
- 326,730 fact prefix context sentences

<!--
TBD: 
- Number of samples
- Data splits
- Task types
- Evaluation scenarios
-->

### Data Format
The dataset is saved as a list of dicts, each of which represents a data instance. 
An example in `CHED` is shown below.

```
{
  "case_id": "6",
  "counterfact_id": "6",
  "prompt": "{}, that was created in",
  "subject": "Anaal Nathrakh",
  "fact_knowledge": "Birmingham",
  "edited_knowledge": "Philadelphia",
  "relation_id": "P740",
  "rephrased_prompt": "In Wardha he came in  close contact with Mahatma Gandhi. Anaal Nathrakh was founded in",
  "locality_prompt": "City of Birmingham Symphony Orchestra, that was created in",
  "locality_ground_truth: "Birmingham,
  "sbj_hop_word: [
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
  "obj_new_hop_word: [
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
```python
# TBD: Basic usage example

# ... example usage
```

### Evaluation Metrics
TBD: 


## Experimental Results

### Analysis 
TBD: 

## How to Cite

If you use CoRE in your research, please cite our paper:

```bibtex

```
