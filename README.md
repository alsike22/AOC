# Deep Autoencoder One-Class Time Series Anomaly Detection
This repository provides the implementation of the _Deep Autoencoder One-Class Time Series Anomaly Detection_ method, called _AOC_ bellow. 

The implementation uses the [Merlion](https://opensource.salesforce.com/Merlion/v1.1.0/tutorials.html) and the [Tsaug](https://tsaug.readthedocs.io/en/stable/notebook/Examples%20of%20augmenters.html) libraries.

## Abstract
> Time-series Anomaly Detection(AD) is widely used in monitoring and security 
pplications in various industries and has become a hot spot in the field of deep learning. 
> Normalityrepresentation-based methods perform well in certain scenarios but may ignore some aspects of the overall normality.
> Feature-extraction-based methods always take a process of pre-training, whose target differs from AD, leading to a decline in AD performance. In this paper, we propose a new AD method called deep Autoencoding One-Class (AOC), which learns features with AutoEncoder(AE). 
> Meanwhile, the normal context vectors from AE are constrained into a hypersphere small enough, similar to one-class methods. 
> With an objective function that optimizes the two assumptions simultaneously, AOC learns various aspects of normality, which is more effective for AD. Experiments on public datasets show that our method outperforms existing baseline approaches.

## Installation
This code is based on `Python 3.8`, all requires are written in `requirements.txt`. Additionally, we should install `saleforce-merlion` and `ts_dataset` as [Merlion](https://github.com/salesforce/Merlion) suggested.

```
git clone https://github.com/salesforce/Merlion.git
cd Merlion
pip install salesforce-merlion
pip install -e Merlion/ts_datasets/
pip install -r requirements.txt
```

## Repository Structure

### `conf`
This directory contains experiment parameters for all models on IOpsCompetition, UCR datasets.

### `models`
Source code of OCSVM, DeepSVDD, TS-TCC and AOC models.

### `results`
Directory where the experiment results and checkpoint are saved.

## Usage
```
python aoc.py --selected_dataset UCR --device cuda --seed 2
python aoc.py --selected_dataset IOpsCompetition --device cuda --seed 2


# Baseline training
# model_name: IsolationForest, RandomCutForest, SpectralResidual, LSTMED, DAGMM, OCSVM, DeepSVDD
python baseline.py --dataset UCR --model <model_name>  --debug

# TS_TCC_AD training
python ts_tcc_main.py --training_mode self_supervised --selected_dataset IOpsCompetition --device cuda --seed 5
python ts_tcc_main.py --training_mode anomaly_detection --selected_dataset IOpsCompetition --device cuda --seed 5
```

## Disclosure
This implementation is based on [Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch), 
[TS-TCC](https://github.com/emadeldeen24/TS-TCC), and [affiliation-metrics](https://github.com/ahstat/affiliation-metrics-py)
