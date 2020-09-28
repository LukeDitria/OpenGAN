# About

Pytorch training code for OpenGAN and the metric feature extractor.
https://arxiv.org/pdf/2003.08074.pdf

Tested with Python 3.5.2, Pytorch 1.1 and Ubuntu 16.04.6.

# Usage
## Datasets
Parent directory of dataset should contain class-specific sub-directories e.g. class_000/, class_001/ etc.
Leading zeros are important if you want the class labels to be sorted in the correct order.

## Feature Extractor Training
Training script is set up to train a ResNet18 model (512-dimensional feature space). This can be changed by altering the train_fe.py file.

Basic usage - replace id with GPU ID and n with the number of training classes (e.g. 82 for Flowers102):
```bash
python3 train_fe.py --data_dir /path/to/dataset --save_dir /path/to/save/directory --gpu_id id  --num_classes n
```
Other training settings (e.g. sigma, maximum training time, batch size etc.) can be seen by running:
```bash
python3 train_fe.py --help
```
or by looking in train_fe.py.

## OpenGAN Training
Basic usage (again, set-up for a Resnet18 feature extractor) - any number of GPU IDs may be entered.
```bash
python3 train_gan.py --save_name experiment_prefix  --data_dir /path/to/dataset --save_dir /path/to/save/directory --fe_model /path/to/feature/extractor/model.pt --gpu_ids id1 id2 id3 id4  --batch_size 48
```
Other training settings (e.g. number of training/novel classes - default is set up for Flowers102 dataset) can be seen by running:
```bash
python3 train_gan.py --help
```
or by looking in train_gan.py.


