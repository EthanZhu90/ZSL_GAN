# ZSL_GAN_CVPR18
code for the paper.

Yizhe Zhu, Mohamed Elhoseiny, Bingchen Liu, Xi Peng and Ahmed Elgammal
"A Generative Adversarial  Approach for Zero-Shot Learning from Noisy Texts", CVPR, 2018


Data:
You can download the dataset [CUBird and NABird](https://drive.google.com/open?id=1YUcYHgv4HceHOzza8OGzMp092taKAAq1)   
Put the uncompressed data to the folder "data"


### CUBird SCS mode
```python
python train_CUB.py --splitmode easy
```

### CUBird SCE mode
```python
python train_CUB.py --splitmode hard
```
### NABird SCS mode
```python
python train_NAB.py --splitmode easy
```

### NABird SCE mode
```python
python train_NAB.py --splitmode easy
```

If you find this implementation or the analysis conducted in our report helpful, please consider citing:

@inproceedings{Yizhe_ZSL_2018,
    Author = {Yizhe Zhu, Mohamed Elhoseiny, Bingchen Liu, Xi Peng and Ahmed Elgammal},
    Title = {A Generative Adversarial  Approach for Zero-Shot Learning from Noisy Texts},
    Booktitle = {CVPR},
    Year = {2018}
}

### TODO:
Add result evaluated on [GRU setting](https://arxiv.org/abs/1707.00600) 

### Updata:
2018/07/17 make code compatable with python2&3   
2018/07/18 merge the train.py and test.py to one file   
2018/07/18 add the experiments of NABird. 
