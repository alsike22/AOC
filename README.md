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
This code is based on `Python 3.8`, all requires are written in `requirements.txt`. Additionally, we should install `saleforce-merlion v1.1.1` and `ts_dataset` as [Merlion](https://github.com/salesforce/Merlion) suggested.

```
git clone https://github.com/salesforce/Merlion.git
cd Merlion
pip install salesforce-merlion==1.1.1
pip install -e Merlion/ts_datasets/
pip install -r requirements.txt
```

The AOC repository already includes the merlion's data loading package `ts_datasets`.
Please unzip the `data/iops_competition/phase2.zip` before running the program.

## Repository Structure

### `conf`
This directory contains experiment parameters for AOC model on IOpsCompetition, UCR datasets.

### `models`
Source code of AOC models.

### `results`
Directory where the experiment results and checkpoint are saved.

## Usage
```
python aoc.py --selected_dataset UCR --device cuda --seed 2
python aoc.py --selected_dataset IOpsCompetition --device cuda --seed 2
```

## Disclosure
This implementation is based on [affiliation-metrics](https://github.com/ahstat/affiliation-metrics-py)
