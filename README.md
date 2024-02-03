This repository highly depends on the <a href="https://github.com/Junjue-Wang/LoveDA">LoveDA</a> and  <a href="https://github.com/lslrh/CPSL">CPSL</a>. We thank the authors for their great work and clean code. Appreciate it!
LoveDA and ISPRS datasets can be downloaded via links  <a href="https://zenodo.org/records/5706578">LoveDA</a> and  <a href="https://github.com/te-shi/MUCSS">ISPRS 2D datasets</a>, respectively

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
#### 1. Compute class distrubution
```
Python generate_class_distribution.py
```
#### 2. Compute prototype for self-training initialization
```
Python calc_prototype.py
```


### Train AST Model
for example transfer setting Rural to Urban, default='ast.2rural'
``` 
python train.py
```
Eval Model on Rural val set
```
python eval.py
```
Test
```
python test.py
```
Submit your test results on [LoveDA Unsupervised Domain Adaptation Challenge](https://codalab.lisn.upsaclay.fr/competitions/424) and you will get your Test score.
