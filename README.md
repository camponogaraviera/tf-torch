<!-- Shields: -->

[![Python](https://img.shields.io/badge/Python-3.11.0-informational)](https://www.python.org/downloads/source/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_CPU-2.17.0-%23FF6F00)](https://www.tensorflow.org/install/source#cpu)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_GPU-2.17.0-%23FF6F00)](https://www.tensorflow.org/install/source#gpu)
[![Torch](https://img.shields.io/badge/Torch-2.7.1-%23EE4C2C)](https://pytorch.org/)
[![Contributions](https://img.shields.io/badge/contributions-welcome-orange?style=flat-square)](https://github.com/camponogaraviera/tf-torch/pulls)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/camponogaraviera/tf-torch/graphs/commit-activity)


<div align='center'>
  <h1> TensorFlow & PyTorch </h1>
  <h1> API Tutorial </h1>
</div>

# About

This repository provides hands-on tutorials about the **Sequential API**, **Functional API**, and **Model Subclassing** in TensorFlow and PyTorch. 

# Table of Contents

- [TensorFlow API Tutorial](tensorflow.ipynb)
- [PyTorch API Tutorial](torch.ipynb)

# Setting Up the Development Environment

> Important: Ensure that the versions of TensorFlow or PyTorch you install are compatible with your Python version and GPU drivers/CUDA versions.

This setup has been tested with an NVIDIA L40S 45GB GPU running driver version 550.127.08, which is backward compatible with earlier CUDA versions (e.g., CUDA 11.8).
  
- Verify GPU availability:

```bash
nvidia-smi
```

## PyTorch Setup

- Create and activate the environment:

```bash
conda env create -f torch_env.yml && conda activate torch-env
```

- Verify the GPU installation:

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
conda env create -f tf_cpu_env.yml && conda activate tf-cpu-env
```

- TensorFlow-GPU:

```bash
conda env create -f tf_gpu_env.yml && conda activate tf-gpu-env
```

- Verify the GPU installation:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output:

```bash
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]
```
