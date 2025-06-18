<!-- Shields: -->

[![Python](https://img.shields.io/badge/Python-3.10-informational)](https://www.python.org/downloads/source/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_CPU-2.14.0-%23FF6F00)](https://www.tensorflow.org/install/source?#cpu)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_GPU-2.9-%23FF6F00)](https://www.tensorflow.org/install/source_windows?#gpu)
[![Torch](https://img.shields.io/badge/Torch-2.2.0-6133BD)](https://pytorch.org/)

<div align='center'>
  <h1> TensorFlow & PyTorch API </h1>
  <h1> Tutorial </h1>
</div>

# Table of Contents

- [tensorflow.ipynb](tensorflow.ipynb): Sequential API, Functional API, OOP implementation, and MNIST example.
- [torch.ipynb](torch.ipynb): Sequential API, Functional API, OOP implementation, and MNIST example.

# Setting up the development environment

- Torch

```bash
conda env create -f torch_env.yml && conda activate torch_env
```

- TF-CPU
  
```bash
conda env create -f tf_cpu_env.yml && conda activate tf_cpu_env
```

- TF-GPU

```bash
conda env create -f tf_gpu_env.yml && conda activate tf_gpu_env
```

Check GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
`>>> [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`
