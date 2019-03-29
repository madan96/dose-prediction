# Dose-prediction

Implementation of the paper Automatic treatment planning based on three-dimensional dose distribution
predicted from deep learning technique ([Link](https://www.ncbi.nlm.nih.gov/pubmed/30383300)) in Tensorflow

## Requirements

Install the requreired python packages using:

```pip install -r requirements.txt```

## Usage

To train the network, run:

```python train.py```

Specify the root directory using:

```python train.py --root_dir /home/user/```

Specify dataset directory using:

```python train.py --dataset_dir train_data```

For the complete list of command line options, run:

```python train.py --help```

## Logs

Loss curve and predicted dose distribution can be viewed from the summary folder present in the root directory using Tensorboard

```tensorboard --logdir=[path_to_summary_folder]```

## Saved Models

Saved models can be accessed from the checkpoint directory
