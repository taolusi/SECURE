# SECURE
This is the implementation of the ACL 2024 Main paper: [**S**ynergetic **E**vent Understanding: A **C**ollaborative Approach to Cross-Doc**u**ment Event Coreference **Re**solution with Large Language Models](https://arxiv.org/abs/2407.02148.).

This codebase builds upon the implementation from [Focus on what matters: Applying Discourse Coherence Theory to Cross Document Coreference](https://github.com/Helw150/event_entity_coref_ecb_plus), with further enhancements and modifications.

## Quick links

* [Overview](#overview)
* [Preparation](#preparation)
  * [Environment](#environment)
  * [Data](#data)
* [Run the model](#run-lm-bff)
  * [Quick start](#quick-start)
  * [Experiments with multiple runs](#experiments-with-multiple-runs)
  * [Without bipartite loss](#without-bipartite-loss)
  * [Joint/Single prompts](#joint-prompt-or-not)
  * [Manual/Concat/Soft prompts](#manual-prompt-or-others)
  * [Few-shot setting](#few-shot-setting)
* [Citation](#citation)

## Overview
![](./model_framework.jpg)

In this work we present SECURE: a collaborative approach for cross-document event coreference resolution, leveraging the capabilities of both a
universally capable LLM and a task-specific
SLM. We formulate our contribution as follow.

1. We design generic tasks to leverage the potential
of LLMs for CDECR, effectively bridging the gap
between the general capabilities of LLMs and the
complex annotation guidelines of specific IE tasks. 
2. We focus on processing each mention individually, which is more efficient compared to existing methods that require handling combinations of mention pairs, resulting in a quadratic increase in processing entries.


## Preparation

### Environment
To run our code, please first ensure you have Python installed. This codebase is tested with Python version 3.10.13. We recommend using [Conda](https://docs.anaconda.com/anaconda/) for managing your Python environment: 
```
conda create -n secure python=3.10.13
conda activate secure
```

To install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

For the usage of spacy, the following command could be helpful:
```
python -m spacy download en_core_web_sm
```

### Data
We conduct experiments on three common datasets: ECB+, GVC and FCC. You can use our provided preprocessed data or construct the datasets from scratch yourself.
- ECB+: This dataset can be downloaded from [http://www.newsreader-project.eu/results/data/the-ecb-corpus/](http://www.newsreader-project.eu/results/data/the-ecb-corpus/).
- RAMS / WIKIEVENTS: We write a script for you for data processing. Run the following commands in the root directory of the repo.

```bash
bash ./data/download_dataset.sh
```  

Please make sure your data folder structure as below.
```bash
data
  ├── ace_eeqa
  │   ├── train_convert.json
  │   ├── dev_convert.json
  │   └── test_convert.json
  ├── RAMS_1.0
  │   └── data
  │       ├── train.jsonlines
  │       ├── dev.jsonlines
  │       └── test.jsonlines
  ├── WikiEvent
  │   └── data
  │       ├── train.jsonl
  │       ├── dev.jsonl
  │       └── test.jsonl
  ├── prompts
  │   ├── prompts_ace_full.csv
  │   ├── prompts_wikievent_full.csv
  │   └── prompts_rams_full.csv
  └── dset_meta
      ├── description_ace.csv
      ├── description_rams.csv
      └── description_wikievent.csv
```

## Run the model

### Quick start
You could simply run PAIE with following commands: 
```bash
bash ./scripts/train_{ace|rams|wikievent}.sh
```
Folders will be created automatically to store: 

1. Subfolder `checkpoint`: model parameters with best dev set result
2. File `log.txt`: recording hyper-parameters, training process and evaluation result
3. File `best_dev_results.log`/`best_test_related_results.log`: showing prediction results of checkpoints on every sample in dev/test set.

You could see hyperparameter setting in `./scripts/train_[dataset].sh` and `config_parser.py`. We give most of hyperparameters a brief explanation in `config_parser.py`.

Above three scripts train models with BART-base. If you want to train models with BART-Large, please change `--model_name_or_path` from `facebook/bart-base` to `facebook/bart-large` **or** run following commands:
```bash
bash ./scripts/train_{ace|rams|wikievent}_large.sh
```

### Experiments with multiple runs

Table.3 of [our paper](https://arxiv.org/pdf/2202.12109.pdf) shows the fluctuation of results due to random seed and other hyperparameters (learning rate mainly). You could run experiments multiple times to get a more stable and reliable results.

```bash
for seed in 13 21 42 88 100
do
    for lr in 1e-5 2e-5 3e-5 5e-5
    do
        bash ./scripts/train_{ace|rams|wikievent}.sh $seed $lr
    done
done
```

Each run will take ~4h so we highly recommend you to execute above command in parallel way.

### Without-bipartite-loss
You could run PAIE without bipartite matching loss by delete the command argument `--bipartite` **or** run following commands:
```bash
bash ./scripts/train_{ace|rams|wikievent}_nobipartite.sh
```

### Joint-prompt-or-not
Unlike multiple prompt strategy in PAIE, you could also prompt argument using template containing only one role (single prompt). Try it by changing `--model_type` from `paie` to `base` and set proper hyperparameters: `--max_span_num`, `--max_dec_seq_length` and `--th_delta`. Alternatively you could run following commands directly with hyperparameters we tuned:
```bash
bash ./scripts/train_{ace|rams|wikievent}_singleprompt.sh
```

### Manual-prompt-or-others
Besides manual prompt, provide another two joint-prompt choices as described in Section 3.2 of  [our paper](https://arxiv.org/pdf/2202.12109.pdf). We concelude them in the following:
1. (Default setting) Manual Prompt: All roles are connected manually with natural language
2. Concatenation Prompt: To concatenate all role names belonging to one event type.
3. Soft Prompt: Following [previous work](https://arxiv.org/abs/2104.06599) about continuous prompt, we connect different roles with learnable, role-specific pseudo tokens.

Run following commands if you want to try Concatenation Prompt:
```bash
bash ./scripts/train_{ace|rams|wikievent}_concatprompt.sh
```

Run following commands if you want to try Soft Prompt:
```bash
bash ./scripts/train_{ace|rams|wikievent}_softprompt.sh
```


### Few-shot-setting
PAIE also performs well under low-annotation scenario. You could try it by set hyperparameters `--keep_ratio` to a number between 0 to 1, which controls the resampling rate from the original training examples. Simply you could also run scripts below:
```bash
KEEP_RATIO=0.2 bash ./scripts/train_{ace|rams|wikievent}_fewshot.sh
```
Note you could adjust the `KEEP_RATIO` value by yourself.

## Citation
Please cite our paper if you use PAIE in your work:
```bibtex
@inproceedings{ma-etal-2022-prompt,
    title = "{P}rompt for Extraction? {PAIE}: {P}rompting Argument Interaction for Event Argument Extraction",
    author = "Ma, Yubo  and
      Wang, Zehao  and
      Cao, Yixin  and
      Li, Mukai  and
      Chen, Meiqi  and
      Wang, Kun  and
      Shao, Jing",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.466",
    doi = "10.18653/v1/2022.acl-long.466",
    pages = "6759--6774",
}
```
conda create -n secure python=3.10.13
conda activate secure
pip install -r requirements.txt