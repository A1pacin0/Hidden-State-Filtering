# Hidden-State-Filtering
This repository is intended to reproduce the experimental results of Hidden-State-Filtering (HSF) for defending against jailbreak attacks. You can check the ```result``` directory to view the experimental outcomes.
## Quick Start

### 1. Install

```shell
pip install -r requirements.txt
```

### 2. Parepare the hidden states

```shell
bash foward.sh
```
Execute the ```forward.sh``` script to obtain the hidden states from the last Decoder Layer of the models.

### 3.Train HSF 
```shell
bash train_fe2+te1_loop.sh
```
Execute the ```train_fe2+te1_loop.sh``` script to train HSF.

### 4. Test HSF
```shell
bash predict_fe2+te1_loop.sh
```
Run the ```predict_fe2+te1_loop.sh``` script to test the HSF model. This will evaluate the accuracy and F1-score of HSF against jailbreak attacks. Note: Ensure that you execute the ```forward.sh``` script first to obtain the hidden states of the jailbreak attack inputs.


