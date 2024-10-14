# 動きによる4D点群データのセグメンテーション

## DataSets
[MANO](https://mano.is.tue.mpg.de/index.html)  
DownloadのRegistrationを使用

## How to Use
1. **Create ndarray data**:
```
$ python ply2ndarray.py <data_path> <out_path>
```
example:  
`$ python ply2ndarray.py "./data/pointclouds/bodyHands_REGISTRATIONS_A01/" "./data/pointclouds/bodyHands_REGISTRATIONS_A01/A01_pc_array.pkl"`

2. **Segmentation**:
```
$ python main_batch.py
```