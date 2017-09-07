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

## Before running any experiment, remove system-wide cache, using
- `sudo sysctl -w vm.drop_caches=3`
- Use `nocache` before running any command


# Experiments

## Fetching runtime differences
Fetched 50 batches each of size: `torch.Size([10, 3, 18, 84, 84])`

### Run 1
```
nocache python data_loader_jpeg.py 
61.415191650390625

nocache python data_loader_gulpio.py 
5.9158337116241455
```

### Run 2
```
nocache python data_loader_jpeg.py 
58.36166548728943

nocache python data_loader_gulpio.py 
6.112927436828613
```
There is roughly 10 times difference in data fetching time, which is also
corroborated by `sudo iotop` DISK READ speed. 

## Training experiments
- Jpeg script: `CUDA_VISIBLE_DEVICES=0 python train_jpeg.py --config configs/config_jpeg.json -g 0`
- GulpIO script: `CUDA_VISIBLE_DEVICES=1 python train_gulp.py --config configs/config_gulpio.json -g 0`

