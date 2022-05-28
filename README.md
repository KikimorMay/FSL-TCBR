# FSL-TCBR

Paper name:
Alleviating the Sampling Bias of Few Shot Data by Removing Projection to the Centroid



## Preprocessing:
For illustration, we use the publicly available algorithm with extracted features.
We use the same backbone network and training strategies as 'S2M2_R'. Please refer to https://github.com/nupurkmr9/S2M2_fewshot for the backbone training.


#### Step1.
The reviewers can find the extracted feature in <https://drive.google.com/open?id=1JtA7p3sDPksvBmOsJuR4EHw9zRHnKurj>   download the miniImagenet.tar.gz

#### Step2.
Create a new file ./checkpoints/miniImagenet/WideResNet28_10_S2M2_R/
Please put the features extracted by the trained 'S2M2_R'  in the file :   

./checkpoints/miniImagenet/WideResNet28_10_S2M2_R/
base_feature.plk
novel_feature.plk
val_feature.plk



#### Step3.

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
python train_cfc.py --cls new --appro_stastic [support/transductive] --n_shot [1/5] 
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
python train_cfc.py --cls new --appro_stastic base_appro --n_shot [1/5] --num_neighbors [30000/15000/10000/5000/1000/500]
```

With varied number of base neighbors, the accuracy is shown in the follow:

| Number of base neighbors | 5-way 1-shot | 5-way 5-shot |
| :----------------------: | :----------: | :----------: |
|          30000           |    67.79     |    84.32     |
|          15000           |    68.06     |    84.49     |
|          10000           |    68.05     |    84.51     |
|           5000           |    67.06     |              |
|           1000           |    65.48     |              |
|           500            |    64.93     |              |




## Visualization

To show the sampling bias aggravated by task centorid is a naturally occurring phenomenon in few-shot learning, as stated in Section 3.3, please

- Run:

```eval
python bias_simulation.ipynb
```

for the visualization of the simulation experiments with varied number of shots and a.