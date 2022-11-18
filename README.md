
# FSL-TCBR

Paper name:
Alleviating the Sampling Bias of Few Shot Data by Removing Projection to the Centroid [NeurIPS 2022, Spotlight]

Slides: five-minute short version [(link)](https://drive.google.com/file/d/10riiWTvE8fnqwtyBJTuCaE2CsRnP9zfL/view?usp=share_link)



## Preprocessing:
For illustration, we use the publicly available algorithm with extracted features.
We use the same backbone network and training strategies as 'S2M2_R'. Please refer to the [link](https://github.com/nupurkmr9/S2M2_fewshot) for the backbone training.


#### Step1.
We provide the extracted feature in the file  ./checkpoints/miniImagenet/WideResNet28_10_S2M2_R/. If it is unusable, please refer to the [link](https://drive.google.com/open?id=1JtA7p3sDPksvBmOsJuR4EHw9zRHnKurj)   provided by the authors of S2M2_R and download the miniImagenet.tar.gz



#### Step2.

Create a  new cache/ file  in the main file



## Evaluate

### baseline

To show get the accuracy of baseline,

- Run:

```
python evaluate_DC_minusEffect.py --cls cosine --n_shot [1/5]
```

1-shot:  64.63 $\pm​$  0.43

5-shot:  83.62 $\pm​$ 0.29



### The proposed TCPR

To get the performance of proposed transformation TCPR with different approximation of the task centroid:

- Run:

```
python evaluate_DC_minusEffect.py --cls new --appro_stastic [support/transductive] --n_shot [1/5] 
```

By using "--appro_stastic support", the approximated task centorid is calculated by the mean of the support data:

1-shot:  66.89 $\pm$ 0.42

5-shot:  84.06 $\pm$ 0.29

By using "--appro_stastic transductive", the approximated task centorid is calculated by the mean of the support and query data:

1-shot:  69.57 $\pm$  0.42

5-shot:  84.75 $\pm​$ 0.29

By using "--appro_stastic base_appro", the approximated task centorid is calculated by the similar base neighbors:

- Run:

```
python evaluate_DC_minusEffect.py --cls new --appro_stastic base_appro --n_shot [1/5] --num_neighbors [30000/15000/10000/5000/1000/500]
```

With varied number of base neighbors, the accuracy is shown in the follow:

| Number of base neighbors | 5-way 1-shot | 5-way 5-shot |
| :----------------------: | :----------: | :----------: |
|          30000           |    67.79     |    84.42     |
|          15000           |    68.06     |    84.49     |
|          10000           |    68.05     |    84.51     |
|           5000           |    67.06     |    84.28     |
|           1000           |    65.48     |    83.50     |
|           500            |    64.93     |    83.24     |



### Performance on Meta-dataset

We provide the extracted novel features on Meta-learning(trained on miniImagenet with S2M2) in the [link](https://drive.google.com/file/d/16EvU3PSqHCLQi3Yv7ZeVSE7C8Bwgrn85/view?usp=share_link). You can download those features and put it into the checkpoints folder.  Please refer the k in the paper.

- Run:

```
python evaluate_DC_minusEffect.py --cls new --appro_stastic [base_appro/support/transductive] --n_shot [1/5] ----num_neighbors [5000/30000/...] --dataset [mini2CUB/mini2dtd/...]
```



## Simulations with Gaussian Distribution 

#### To show the sample bias aggravated by task centorid is a naturally occurring phenomenon in few-shot learning, as stated in Section 3.3, please

- Run:

```eval
python gaussian_simulation.py --a [0.5/1/2/3] --n_shot [1/3/5/10]
```

for the visualization of the simulation experiments with varied number of shots and a.


#### To show the task centroid is existing in the real data, as shown in the Figure 1, please 

- Run:

```
python real_data_simulation.py
```

- or Run:
```
python evaluate_DC_minusEffect.py --draw_selected_classes True --cls [cosine/new]
```
to get the accuracy with distance as first.

## References 

This code is based on the implementations of  [s2m2](https://github.com/nupurkmr9/S2M2_fewshot) and [Few-shot Distribution Calibration](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration) 



