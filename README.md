# GulpIO-benchmarks
Contains scripts producing performance benchmarks for GulpIO using PyTorch

# Requirements
- Python 3.x
- PyTorch (0.2.0.post4)
- GulpIO

# Steps to reproduce

## Download Jester dataset
Follow instructions on - https://www.twentybn.com/datasets/jester

## Gulping the dataset (need to build GulpIO first)
```
gulp_20bn_csv_jpeg  --num_workers=8 ../GulpIO-benchmarks/csv_files/jester-v1-train.csv /hdd/20bn-datasets/20bn-jes
ter-v1 /hdd/20bn-datasets/20bn-jester-v1-gulpio/train/

gulp_20bn_csv_jpeg ../GulpIO-benchmarks/csv_files/jester-v1-validation.csv /hdd/20bn-datasets/20bn-jester-v1 /hdd/
20bn-datasets/20bn-jester-v1-gulpio/validation/
```

Should obtain something like this:
```
37G     20bn-jester-v1
30G     20bn-jester-v1-gulpio
67G     total
```
