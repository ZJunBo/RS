This repository highly depends on the <a href="https://github.com/Junjue-Wang/LoveDA">LoveDA</a> and  <a href="https://github.com/lslrh/CPSL">CPSL</a>. We thank the authors for their great work and clean code. Appreciate it!


## Getting Started

#### Requirements:
- pytorch >= 1.7.0
- python >=3.6
- pandas >= 1.1.5
### Prepare LoveDA Dataset

```bash
ln -s </path/to/LoveDA> ./LoveDA
```


### Prepare the necessary files for self-training
#### 1. Python generate_class_distribution.py
#### 2. Python calc_prototype.py
```bash
mkdir -vp ./log/
mv ./CBST_2Urban.pth ./log/CBST_2Urban.pth
```

#### 3. Evaluate on Urban test set
```bash 
bash ./scripts/predict_cbst.sh
```
Submit your test results on [LoveDA Unsupervised Domain Adaptation Challenge](https://codalab.lisn.upsaclay.fr/competitions/424) and you will get your Test score.

### Train CBST Model
From Rural to Urban
```bash 
bash ./scripts/train_cbst.sh
```
Eval CBST Model on Urban val set
```bash
bash ./scripts/eval_cbst.sh
```
