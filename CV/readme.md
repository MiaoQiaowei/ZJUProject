# CV Homework

Self-developed CV Mnist Project

## Model
- CNN(Resnet-18)
- SVM
- BoW

## Run
```
python runner.py
```

## Dataset
Mnist
|域|0  |1  |2  |3  |4  |5  |6  |7  |8  |9  |
|:---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|train|2274|2624|2238|2380|2240|2035|2264|2380|2185|2304|
|test |100|100|100|100|100|100|100|100|100|100|100|100|


## Result

|模型|ACC|备注|
|:--:|:--:|:--:|
|CNN|30.7|lr=0.05,, epoch=3|
|SVM|97.3|无|
|BoW|12.1|SIFT|