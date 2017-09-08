# GulpIO-benchmarks

Scripts to run performance benchmarks for GulpIO using PyTorch.

# Requirements

- Python 3.x
- PyTorch (0.2.0.post4)
- GulpIO (latest)
- nocache

# Steps to reproduce

## Download *Jester* dataset

During these benchmarks, we will use the publicly available 20BN-jester
dataset, a collection of around 150k short videos of people performing hand
gestures. The dataset is available from: https://www.twentybn.com/datasets/jester

Be sure to grab both the CSV files with the labels and the actual data.

## *Gulp* the dataset

Before you begin, you must *gulp* the dataset. You can use the command-line
utilities provided with the GulpIO package. Replace the paths accordingly for
your machine.

```
$ gulp_20bn_csv_jpeg --num_workers=8 csv_files/jester-v1-train.csv /hdd/20bn-datasets/20bn-jes
ter-v1 /hdd/20bn-datasets/20bn-jester-v1-gulpio/train/

$ gulp_20bn_csv_jpeg --num_workers=8 csv_files/jester-v1-validation.csv /hdd/20bn-datasets/20bn-jester-v1 /hdd/
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


## Cleaning the filesystem cache

Before running any experiment, drop the filesystem cache, using: `sudo sysctl
-w vm.drop_caches=3`. Since we are benchmarking disk-reads this step is
essential to obtain accuerate results.

When running an experiment, Use the `nocache` command line utility before
executing any command. This should ensure that the filesystem cache is byassed
and that you can run multiple times and still obtain accurate results. You can
read more about `nocache` here: https://github.com/Feh/nocache

# Experiments

All the resultes reported here were run on a desktop dual GPU sytem with the
following specs:

* 2x GTX 1080 Ti
* Hexacore  Intel i7-6850K Processor
* 128 GB RAM
* 3TB Western Digital disk
* MSI x99A Motherboard

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
corroborated by `sudo iotop` `DISK READ` speed.

## Training experiments
- Jpeg script: `CUDA_VISIBLE_DEVICES=0 python train_jpeg.py --config configs/config_jpeg.json -g 0`
### Epoch 1 result

- GulpIO script: `CUDA_VISIBLE_DEVICES=1 python train_gulp.py --config configs/config_gulpio.json -g 0`
### Epoch 1 result
```
(torch) root@7396071ace7e:/home/rgoyal/GulpIO-benchmarks# CUDA_VISIBLE_DEVICES=0,1 python train_gulp.py --config configs/config_gulpio.json -[13/511]
=> active GPUs: 0,1
=> Output folder for this run -- jester_conv_example
 > Using 10 processes for data loader.
 > Training is getting started...
 > Training takes 999999 epochs.
 > Current LR : 0.001
Epoch: [0][0/1852]      Time 12.410 (12.410)    Data 9.079 (9.079)      Loss 3.2856 (3.2856)    Prec@1 9.375 (9.375)    Prec@5 17.188 (17.188)
Epoch: [0][100/1852]    Time 0.457 (0.854)      Data 0.000 (0.429)      Loss 2.5770 (3.0944)    Prec@1 26.562 (12.655)  Prec@5 68.750 (37.036)
Epoch: [0][200/1852]    Time 0.456 (0.818)      Data 0.000 (0.432)      Loss 2.6250 (2.8789)    Prec@1 25.000 (18.043)  Prec@5 56.250 (47.528)
Epoch: [0][300/1852]    Time 0.458 (0.792)      Data 0.000 (0.419)      Loss 2.4090 (2.7366)    Prec@1 34.375 (21.522)  Prec@5 68.750 (53.265)
Epoch: [0][400/1852]    Time 0.464 (0.770)      Data 0.000 (0.398)      Loss 2.0964 (2.6066)    Prec@1 28.125 (24.762)  Prec@5 76.562 (57.988)
Epoch: [0][500/1852]    Time 0.979 (0.751)      Data 0.770 (0.377)      Loss 2.0077 (2.4919)    Prec@1 45.312 (27.561)  Prec@5 81.250 (61.642)
Epoch: [0][600/1852]    Time 1.648 (0.733)      Data 1.448 (0.358)      Loss 2.0193 (2.3956)    Prec@1 39.062 (30.135)  Prec@5 79.688 (64.655)
Epoch: [0][700/1852]    Time 0.467 (0.714)      Data 0.103 (0.339)      Loss 1.8151 (2.3129)    Prec@1 39.062 (32.280)  Prec@5 85.938 (67.005)
Epoch: [0][800/1852]    Time 0.465 (0.697)      Data 0.000 (0.319)      Loss 1.3762 (2.2394)    Prec@1 59.375 (34.131)  Prec@5 90.625 (68.959)
Epoch: [0][900/1852]    Time 0.464 (0.682)      Data 0.000 (0.300)      Loss 1.6394 (2.1730)    Prec@1 43.750 (35.972)  Prec@5 85.938 (70.651)
Epoch: [0][1000/1852]   Time 0.463 (0.665)      Data 0.000 (0.280)      Loss 1.5073 (2.1184)    Prec@1 45.312 (37.431)  Prec@5 85.938 (72.056)
Epoch: [0][1100/1852]   Time 0.464 (0.648)      Data 0.017 (0.258)      Loss 1.3807 (2.0655)    Prec@1 54.688 (38.816)  Prec@5 95.312 (73.348)
Epoch: [0][1200/1852]   Time 0.461 (0.633)      Data 0.000 (0.238)      Loss 1.5314 (2.0186)    Prec@1 50.000 (40.058)  Prec@5 85.938 (74.491)
Epoch: [0][1300/1852]   Time 0.490 (0.620)      Data 0.000 (0.220)      Loss 1.7921 (1.9743)    Prec@1 51.562 (41.244)  Prec@5 81.250 (75.533)
Epoch: [0][1400/1852]   Time 0.463 (0.609)      Data 0.000 (0.204)      Loss 1.3163 (1.9360)    Prec@1 56.250 (42.276)  Prec@5 92.188 (76.410)
Epoch: [0][1500/1852]   Time 0.463 (0.600)      Data 0.000 (0.191)      Loss 1.2276 (1.8984)    Prec@1 64.062 (43.260)  Prec@5 87.500 (77.213)
Epoch: [0][1600/1852]   Time 0.464 (0.591)      Data 0.000 (0.179)      Loss 1.2961 (1.8634)    Prec@1 59.375 (44.140)  Prec@5 87.500 (77.967)
Epoch: [0][1700/1852]   Time 0.464 (0.584)      Data 0.000 (0.168)      Loss 1.3273 (1.8323)    Prec@1 60.938 (45.009)  Prec@5 89.062 (78.635)
Epoch: [0][1800/1852]   Time 0.469 (0.577)      Data 0.000 (0.159)      Loss 1.2787 (1.8019)    Prec@1 57.812 (45.827)  Prec@5 90.625 (79.255)
 > Time taken for this 1 train epoch = 1062.678290605545
Test: [0/232]   Time 6.247 (6.247)      Loss 1.5232 (1.5232)    Prec@1 53.125 (53.125)  Prec@5 82.812 (82.812)
Test: [100/232] Time 0.181 (0.406)      Loss 1.1259 (1.3052)    Prec@1 70.312 (59.746)  Prec@5 95.312 (89.975)
Test: [200/232] Time 0.182 (0.390)      Loss 1.2444 (1.3056)    Prec@1 59.375 (59.600)  Prec@5 92.188 (90.003)
 * Prec@1 59.654 Prec@5 90.011
 > Time taken for this 1 validation epoch = 88.27695989608765

```

# License

Copyright (c) 2017 Twenty Billion Neurons GmbH, Berlin, Germany

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
