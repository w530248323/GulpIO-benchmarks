# GulpIO-benchmarks

Scripts to run performance benchmarks for GulpIO using PyTorch.

# Requirements
- Python 3.x
- PyTorch (0.2.0.post4)
- GulpIO (latest)
- nocache

# Steps to reproduce

## Download *Jester* dataset

During these benchmarks, we will use the publically available 20BN-jester
dataset, a collection of around 150k short videos of people performing hand
gestures. The dataset is available from: https://www.twentybn.com/datasets/jester

Be sure to grab both the CSV files with the labels and the actual data.

## Gulp the dataset

Before you begin, you must *gulp* the dataset. You can use the command-line
utilities provided with the GulpIO package. Replace the paths accordingly for
your machine.

```
$ gulp_20bn_csv_jpeg  --num_workers=8 csv_files/jester-v1-train.csv /hdd/20bn-datasets/20bn-jes
ter-v1 /hdd/20bn-datasets/20bn-jester-v1-gulpio/train/

$ gulp_20bn_csv_jpeg csv_files/jester-v1-validation.csv /hdd/20bn-datasets/20bn-jester-v1 /hdd/
20bn-datasets/20bn-jester-v1-gulpio/validation/
```

You should obtain something like this:

```
$ du -sch /hdd/20bn-datasets/
37G     20bn-jester-v1
30G     20bn-jester-v1-gulpio
67G     total
```

The *gulped* dataset is smaller because...


## Before running any experiment, remove system-wide cache, using
- `sudo sysctl -w vm.drop_caches=3`
- Use the `nocache` command line utility before running any command.


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

