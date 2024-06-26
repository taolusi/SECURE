# SECURE
This is the implementation of the ACL 2024 Main paper: [**S**ynergetic **E**vent Understanding: A **C**ollaborative Approach to Cross-Doc**u**ment Event Coreference **Re**solution with Large Language Models](https://arxiv.org/abs/2407.02148.).

This codebase builds upon the implementation from [Focus on what matters: Applying Discourse Coherence Theory to Cross Document Coreference](https://github.com/Helw150/event_entity_coref_ecb_plus), with further enhancements and modifications.

## Codebase Release Progress

We are gradually releasing materials related to our paper. The release includes the following:

- [x] Code for data preparation on ECB+
- [ ] Code for model training and evaluation on ECB+
- [ ] Code for LLM summary generation on ECB+
- [ ] Trained checkpoints on ECB+
- [ ] Original and generated data on ECB+
- [ ] The same steps above apply to GVC and FCC as well

Beyond our initial release, we are also planning to release the following;
however, they are subject to change (in terms of the release date and the content):

- [ ] Migrations of backbond model from RoBERTa to Longformer/CDLM

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
- ECB+: run the following command to construct the datasets:

```bash
bash ./data/download_dataset.sh
```  