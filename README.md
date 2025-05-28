# Temperature-Annealed Boltzmann Generators

This repository contains the code to reproduce the experiments of our paper
[Temperature-Annealed Boltzmann Generators](https://arxiv.org/abs/2501.19077)
(TA-BG) presented at ICML 2025.

Temperature-Annealed Boltzmann Generators leverage the fact that reverse KLD
training does not suffer from mode collapse when targeting a sufficiently high
temperature. After reverse KLD pre-training, we anneal the distribution of the
Boltzmann generator iteratively using importance sampling. We apply this
methodology to the sampling of the Boltzmann distribution of three increasingly
complex molecular systems. Details can be found in our
[publication](https://arxiv.org/abs/2501.19077).

## Requirements

An environment with all dependencies can be installed in the following way:

```bash
conda env create -f environment.yaml
```

Since we use [weights and biases](https://wandb.ai/) to track experiments, you first need 
to login to your account:

```bash
wandb login
```

## Downloading ground truth datasets
To evaluate the trained Boltzmann generators, ground truth datasets are needed.
Ground truth datasets obtained from molecular dynamics simulations can be downloaded
from Zenodo as a zip archive: https://doi.org/10.5281/zenodo.15526429

Place the content of the `datasets` folder contained in the zip archive in `./annealed_bg/data/`.

## Running the experiments
The experiments presented in our paper can be performed in the following way:

```bash
conda activate annealed_bg
cd annealed_bg/
python train.py -cd configs/paper/<system_name>/ -cn <config_name>
```

TA-BG experiments are performed in two stages. First, a reverse KLD experiment
at elevated temperature needs to be performed (using, e.g.
`./configs/paper/aldp/rev_kld_1200K.yaml`). The annealing is then performed in a
separate experiment, where the checkpoint from the pre-training is used (set
`config.training.checkpoint_path` to a checkpoint from the pre-training experiment).