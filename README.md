# Estimating Conditional Mutual Information for Dynamic Feature Selection
This paper presents DIME (**di**scriminative **m**utual information **e**stimation), a new modeling approach for dynamic feature selection by estimating the conditional mutual information in a discriminative fashion. The implementation was done using [PyTorch Lightning](https://www.pytorchlightning.ai/index.html). 

## Usage
The ```experiments\``` directory contains subdirectories for each of the datasets used. In each of the subdirectories, the ```greedy_cmi_estimation_pl.py``` file can be run to jointly train the value network and the predictor network as described in the paper. Each subdirectory also contains a ```*.ipynb``` jupyter notebook to evaluate the trained networks using different stopping criteria.
