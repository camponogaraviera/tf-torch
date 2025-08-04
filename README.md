<!-- Shields: -->

[![Python](https://img.shields.io/badge/Python-3.11.0-informational)](https://www.python.org/downloads/source/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_CPU-2.17.0-%23FF6F00)](https://www.tensorflow.org/install/source#cpu)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_GPU-2.17.0-%23FF6F00)](https://www.tensorflow.org/install/source#gpu)
[![Torch](https://img.shields.io/badge/Torch-2.7.1-6133BD)](https://pytorch.org/)

<div align='center'>
  <h1> TensorFlow & PyTorch </h1>
  <h1> API Tutorial </h1>
</div>

# Table of Contents

- [tensorflow.ipynb](tensorflow.ipynb): Sequential API, Functional API, OOP implementation, and MNIST example.
- [torch.ipynb](torch.ipynb): Sequential API, Functional API, OOP implementation, and MNIST example.

# Setting up the development environment

Note: One should ensure that the versions of TensorFlow or PyTorch are compatible with both the Python version and the CUDA version supported by the system's GPU drivers. The following install works for an NVIDIA L40S 45GB GPU with Driver version 550.127.08, which is backward-compatible with earlier CUDA versions (e.g., 11.8).

- Check GPUs:

```bash
nvidia-smi
```

- PyTorch Env.:

```bash
conda env create -f torch_env.yml && conda activate torch-env
```

Check installation:

```bash
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

```bash
GPU 0: NVIDIA L40S
GPU 1: NVIDIA L40S
```

- TensorFlow-CPU Env.:

```bash
conda env create -f tf_cpu_env.yml && conda activate tf-cpu-env
```

- TensorFlow-GPU Env.:

```bash
conda env create -f tf_gpu_env.yml && conda activate tf-gpu-env
```

Check installation:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

`[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]`
