# README
This repository implements the following paper:
```
TitaNet: Neural Model for speaker representation with 1D Depth-wise separable convolutions and global context,
Nithin Rao Koluguri, Taejin Park, Boris Ginsburg,
https://ieeexplore.ieee.org/document/9746806.
```

## Data training
Data training uses from open source [Vietnamese Speaker Verification](https://www.kaggle.com/datasets/dungnasa10/vietnamese-speaker-verification). It contains about 25 hours audios, which are crawled from 1015 vietnamese youtube channels.
This data is not verified by human, so it's quality is low
## Data testing
Data testing is took from [VLSP 2021](https://jcsce.vnu.edu.vn/index.php/jcsce/article/view/333), task Speaker verification.
The data is built from VLSP 2020, and crawled from youtube. Each sample includes a pair audios, and label is it the same speaker.
Dataset is split into 3 parts
* Public dataset: Most samples are taken from VLSP 2020 dataset, which is clean
* Private dataset 1: Samples are taken from youtube audios, which have varied background environments including inaudible chatter, laughs, street noise, school, music,....
* Private dataset 2: It includes most hard samples, for testing robustness of systems.
## Result
Use Equal Error Rate (EER) metric

|Dataset|EER|
|-------|---|
|Public dataset|7.36%|
|Private dataset 1|7.92%|
|Private dataset 2|9.76%|
# References
```
TitaNet: Neural Model for speaker representation with 1D Depth-wise separable convolutions and global context,
Nithin Rao Koluguri, Taejin Park, Boris Ginsburg,
https://ieeexplore.ieee.org/document/9746806.
```
```commandline
VLSP 2021 - SV challenge: Vietnamese Speaker Verification
in Noisy Environments
https://jcsce.vnu.edu.vn/index.php/jcsce/article/view/333
```
```commandline
Vietnamese Speaker Verification
https://www.kaggle.com/datasets/dungnasa10/vietnamese-speaker-verification
```