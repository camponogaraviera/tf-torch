<!-- Logos -->

<a target="_blank" href="https://www.tensorflow.org/"><img width="145" src="https://github.com/camponogaraviera/logos/blob/main/assets/tensorflow.svg" /></a>
&nbsp;
<a href="https://pytorch.org/" target="_blank" rel="noopener noreferrer"><img src="https://github.com/camponogaraviera/logos/blob/main/assets/pytorch.png" width="110"></a>

<!-- Badges -->

[![Python](https://img.shields.io/badge/Python-3.11.0-informational)](https://www.python.org/downloads/source/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_CPU-2.17.0-%23FF6F00)](https://www.tensorflow.org/install/source#cpu)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_GPU-2.17.0-%23FF6F00)](https://www.tensorflow.org/install/source#gpu)
[![Torch](https://img.shields.io/badge/Torch-2.7.1-%23EE4C2C)](https://pytorch.org/)
[![Contributions](https://img.shields.io/badge/contributions-welcome-orange?style=flat-square)](https://github.com/camponogaraviera/tf-torch/pulls)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/camponogaraviera/tf-torch/graphs/commit-activity)

> Originally implemented in 2024. Minor cleanup and package updates in 2026.

<!-- Title -->
<div align='center'>
  <h1> TensorFlow & PyTorch </h1>
  <h1> API Tutorial </h1>
</div>

# About

Hands-on tutorials on **Sequential API**, **Functional API**, and **Model Subclassing** in TensorFlow, and equivalent modeling approaches in PyTorch, including data preprocessing and model training.

# Table of Contents

- [TensorFlow API Tutorial](tensorflow.ipynb)
- [PyTorch API Tutorial](torch.ipynb)

# Setting Up the Development Environment

> Important: ensure that your NVIDIA GPU driver is compatible with the CUDA version bundled with the TensorFlow/PyTorch installed.
> 
> This repository has been tested with an L40S 48GB GPU with driver version 550.127.08.

## PyTorch Setup

- Create and activate the environment:

```bash
conda env create -f torch_env.yml && conda activate torch
```

Check GPU:

```bash
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

Expected output:
  
```bash
GPU 0: NVIDIA L40S
GPU 1: NVIDIA L40S
```

## TensorFlow Setup

- TensorFlow-CPU:

```bash
conda env create -f tf_cpu_env.yml && conda activate tf-cpu
```

- TensorFlow-GPU:

```bash
conda env create -f tf_gpu_env.yml && conda activate tf-gpu
```

Check GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output:

```bash
>>> [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
```
