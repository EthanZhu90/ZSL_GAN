# ZSL_GAN
code for the conference and the journal versions of the paper:

Yizhe Zhu, Mohamed Elhoseiny, Bingchen Liu, Xi Peng and Ahmed Elgammal
"A Generative Adversarial  Approach for Zero-Shot Learning from Noisy Texts", CVPR, 2018


## Processed feature Data:
You can download the dataset [CUBird and NABird](https://drive.google.com/open?id=1YUcYHgv4HceHOzza8OGzMp092taKAAq1)   
Put the uncompressed data to the folder "data"

## Raw wikipedia article data:
Raw wikipedia article data of CUBird and NABird, as well as detailed merging information of NABird,  can be obtained [here](https://drive.google.com/open?id=0B_8vkk7CF-pwckxLQTVkcDBadGc).


## Reproduce results 
#### CUBird SCS mode && SCE mode
```shell
python train_CUB.py --splitmode easy
python train_CUB.py --splitmode hard
```

#### NABird SCS mode && SCE mode
```shell
python train_NAB.py --splitmode easy
python train_NAB.py --splitmode hard
```

## Results evaluated on [GBU setting](https://arxiv.org/abs/1707.00600)[1] 

Download the [data](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/) and uncompress it to the folder 'data/GBU'.
```shell
python train_GBU.py --dataset CUB1 --preprocessing --z_dim 100
python train_GBU.py --dataset AWA1 --preprocessing --z_dim 10
python train_GBU.py --dataset AWA2 --preprocessing --z_dim 10
python train_GBU.py --dataset APY --preprocessing --z_dim 10
python train_GBU.py --dataset SUN --preprocessing --z_dim 10
python train_GBU.py --dataset FLO --preprocessing --z_dim 100
```


| Model      |   AwA1 T1    |    u    |    s    |    H    |   CUB T1    |    u    |    s    |    H    |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|
| DAP [2]      |   44.1  |   0.0   |   **88.7**  |   0.0   |   40.0  |   1.7   |   67.9  |   3.3   |
| CONSE [3]     |   45.6  |   0.4   |   88.6  |   0.8   |   34.3  |   1.6   |   **72.2**  |   3.1   |
| SSE [4]       |   60.1  |   7.0   |   80.5  |   12.9  |   43.9  |   8.5   |   46.9  |   14.4  |
| DEVISE [5]    |   54.2  |   13.4  |   68.7  |   22.4  |   52.0  |   23.8  |   53.0  |   32.8  |
| SJE [6]       |   65.6  |   11.3  |   74.6  |   19.6  |   53.9  |   23.5  |   59.2  |   33.6  |
| LATEM [7]     |   55.1  |   7.3   |   71.7  |   13.3  |   49.3  |   15.2  |   57.3  |   24.0  |
| ESZSL [8]     |   58.2  |   6.6   |   75.6  |   12.1  |   53.9  |   12.6  |   63.8  |   21.0  |
| ALE [9]       |   59.9  |   16.8  |   76.1  |   27.5  |   54.9  |   23.7  |   62.8  |   34.4  |
| SYNC [10]      |   54.0  |   8.9   |   87.3  |   16.2  |   55.6  |   11.5  |   70.9  |   19.8  |
| SAE [11]       |   53.0  |   1.8   |   77.1  |   3.5   |   33.3  |   7.8   |   54.0  |   13.6  |
| DEM [12]        | **68.4** | **32.8** | 84.7  |  47.3 | 51.7  |   19.6  |  57.9  |  29.2 |
| **GAZSL (OURS)** | 68.2 | 29.6 | 84.2  |  43.8 | **55.8**  |   **31.68**  |  61.34  |  **41.78** |

| Model      |   AwA2 T1    |    u    |    s    |    H    |   aPY T1    |    u    |    s    |    H    |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|
| DAP [2]      |   46.1  |   0.0    |   84.7  |   0.0   |   33.8  |   4.8   |   78.3  |   9.0   |
| CONSE [3]     |   44.5  |   0.5   | **90.6**|   1.0   |   26.9  |   0.0   |**91.2** |   0.0   |
| SSE [4]       |   61.0  |   8.1   |   82.5  |   14.8  |   34.0  |   0.2   |   78.9  |   0.4   |
| DEVISE [5]    |   59.7  |   17.1  |   74.7  |   27.8  |   39.8  |   4.9   |   76.9  |   9.2   |
| SJE [6]       |   61.9  |   8.0   |   73.9  |   14.4  |   32.9  |   3.7   |   55.7  |   6.9   |
| LATEM [7]     |   55.8  |   11.5  |   77.3  |   20.0  |   35.2  |   0.1   |   73.0  |   0.2   |
| ESZSL [8]     |   58.6  |   5.9   |   77.8  |   11.0  |   38.3  |   2.4   |   70.1  |   4.6   |
| ALE [9]       |   62.5  |   14.0  |   81.8  |   23.9  |   39.7  |   4.6   |   73.7  |   8.7   |
| SYNC [10]     |   46.6  |   10.0  |   90.5  |   18.0  |   23.9  |   7.4   |   66.3  |   13.3  |
| SAE [11]      |   54.1  |   1.1   |   82.2  |   2.2   |   8.3   |   0.4   |   80.9  |   0.9   |
| DEM [12]           | 67.1 | 30.5 | 86.4 | **45.1 | 35.0  | 11.1|  75.1   |19.4 |
| **GAZSL (OURS)** | **70.2** | **35.4** | **86.9**  |  **50.3**  | **41.13**  |   **14.17**  |  78.63  |  **24.01**|


| Model      |   SUN T1    |    u    |    s    |    H    |    FLO T1   |    u    |    s    |    H    |  
|------------|---------|---------|---------|---------| ------------|---------|---------|---------|
| DAP [2]      |   39.9  |   4.2   |   25.1  |   7.2   | 
| CONSE [3]     |   38.8  |   6.8  |   39.9  |   11.6   |  
| SSE [4]       |   51.5 |   2.1  |   36.4 |   4.0  |   
| DEVISE [5]    |   56.5  |   16.9  |   27.4  |   20.9  |   
| SJE [6]       |   53.7  |   14.7  |   30.5  |   19.8  |  
| LATEM [7]     |   55.3  |   14.7  |   28.8  |   19.5  |  
| ESZSL [8]     |   54.5  |   11.0   |  27.9  |   15.8  |   
| ALE [9]       |   58.1  |   **21.8**  |   33.1  |   **26.3**  |   
| SYNC [10]      |   56.3  |   7.9   |   **43.3**  |   13.4  |  
| SAE [11] |  40.3  | 8.8 |18.0| 11.8  |  
| DEM [12]   | **61.9** | 20.5 | 34.3 |  25.6 | 
| **GAZSL (OURS)** | 61.3 | 22.1 | 39.3 |  **28.3** |60.5| 28.1|77.4|41.2|

**Note:** The results of work **[2-11]** are copied from TABLE6 in **[1]**. The results of work **[12]** are obtained from the authors' [report](https://github.com/lzrobots/DeepEmbeddingModel_ZSL). Thanks to [Li Zhang](http://www.robots.ox.ac.uk/~lz/) for the template of README.
### Citation
If you find this implementation or the analysis conducted in our report helpful, please consider citing:
```
@inproceedings{Yizhe_ZSL_2018,  
    Author = {Yizhe Zhu, Mohamed Elhoseiny, Bingchen Liu, Xi Peng and Ahmed Elgammal},  
    Title = {A Generative Adversarial  Approach for Zero-Shot Learning from Noisy Texts},  
    Booktitle = {CVPR},  
    Year = {2018}  
}
```
### TODO:
* ~~Add result evaluated on [GBU setting](https://arxiv.org/abs/1707.00600)~~ 

### Update:
* 2018/10/22 add AUC Score for CUBird and NABird
* 2018/07/23 add results on GBU setting 
* 2018/07/18 add the experiments of NABird. 
* 2018/07/18 merge the train.py and test.py to one file  
* 2018/07/17 make code compatable with python2&3   
 
 
## References
- [1] [Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://arxiv.org/abs/1707.00600).
  Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata.
  arXiv, 2017.
- [2] [Attribute-Based Classification forZero-Shot Visual Object Categorization](https://cvml.ist.ac.at/papers/lampert-pami2013.pdf).
  Christoph H. Lampert, Hannes Nickisch and Stefan Harmeling.
  PAMI, 2014.
- [3] [Zero-Shot Learning by Convex Combination of Semantic Embeddings](https://arxiv.org/abs/1312.5650).
  Mohammad Norouzi, Tomas Mikolov, Samy Bengio, Yoram Singer, Jonathon Shlens, Andrea Frome, Greg S. Corrado, Jeffrey Dean.
  arXiv, 2013.
- [4] [Zero-Shot Learning via Semantic Similarity Embedding](https://arxiv.org/abs/1509.04767).
  Ziming Zhang, Venkatesh Saligrama.
  ICCV, 2015.
- [5] [DeViSE: A Deep Visual-Semantic Embedding Model](http://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf).
  Andrea Frome*, Greg S. Corrado*, Jonathon Shlens*, Samy BengioJeffrey Dean, Marc’Aurelio Ranzato, Tomas Mikolov.
  NIPS, 2013.
- [6] [Evaluation of Output Embeddings for Fine-Grained Image Classification](https://arxiv.org/abs/1409.8403).
  Zeynep Akata, Scott Reed, Daniel Walter, Honglak Lee, Bernt Schiele.
  CVPR, 2015.
- [7] [Latent Embeddings for Zero-shot Classification](https://arxiv.org/abs/1603.08895).
  Yongqin Xian, Zeynep Akata, Gaurav Sharma, Quynh Nguyen, Matthias Hein, Bernt Schiele
  CVPR, 2016.
- [8] [An embarrassingly simple approach to zero-shot learning](http://proceedings.mlr.press/v37/romera-paredes15.pdf).
  Bernardino Romera-Paredes, Philip H. S. Torr.
  ICML, 2015.
- [9] [Label-Embedding for Image Classification](https://arxiv.org/abs/1503.08677).
  Zeynep Akata, Florent Perronnin, Zaid Harchaoui, Cordelia Schmid.
  PAMI, 2016.
- [10] [Synthesized Classifiers for Zero-Shot Learning](https://arxiv.org/abs/1603.00550).
  Soravit Changpinyo, Wei-Lun Chao, Boqing Gong, Fei Sha.
  CVPR, 2016.
- [11] [Semantic Autoencoder for Zero-Shot Learning](https://arxiv.org/abs/1704.08345).
  Elyor Kodirov, Tao Xiang, Shaogang Gong.
  CVPR, 2017.
- [12] [Learning a Deep Embedding Model for Zero-Shot Learning](https://arxiv.org/abs/1611.05088).
  Li Zhang, Tao Xiang, Shaogang Gong
  CVPR, 2017.
