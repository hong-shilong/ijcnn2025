# A Multi-Scale Feature Fusion Framework Integrating Frequency Domain and Cross-View Attention for Dual-View X-ray Security Inspections

## Project Overview
This repository contains the official implementation of the paper *"A Multi-Scale Feature Fusion Framework Integrating Frequency Domain and Cross-View Attention for Dual-View X-ray Security Inspections"*. The framework is designed to improve the accuracy and robustness of dual-view X-ray security inspection systems by integrating frequency-domain features and cross-view attention mechanisms for effective feature fusion.

The framework consists of three core modules:
- **Frequency Domain Interaction Module (FDIM)**: Enhances frequency-domain features through Fourier transform.
- **Multi-Scale Cross-View Feature Enhancement (MSCFE)**: Utilizes cross-view attention mechanisms to improve feature interactions.
- **Convolutional Attention Fusion Module (CAFM)**: Combines channel attention with depthwise separable convolutions for efficient feature fusion.

Experimental results demonstrate that our method outperforms state-of-the-art approaches, particularly in complex scenarios involving occlusions and object stacking.

## Paper
The full paper is available at: [TODO](https://anonymous.4open.science/r/ijcnn2025-C56D/).

## Experimental Setup
Experiments were conducted on a system with two RTX 4090 GPUs, using CUDA 12.4 and PyTorch 2.5.1. The dataset was split into training, validation, and test sets with a ratio of 7:2:1. We used the AdamW optimizer and trained for 60 epochs. The batch size was 64, and 32 workers were used for data loading to maximize throughput. A warmup strategy was employed, and the learning rate was decayed using the CosineAnnealingLR scheduler.

## Model Performance

### Comparison of Model Performance Across Different Backbones

| Model                        | FLOPs  | Params  | Val\_mAP | Test\_mAP |
|------------------------------|--------|---------|----------|-----------|
| ResNet50-dual                 | 10.801G| 31.929M | 0.8093   | 0.7895    |
| +AHCR                         | 12.405G| 34.882M | 0.8249   | 0.8073    |
| +Ours                         | 14.200G| 48.302M | **0.8533**| **0.8498**|
| ResNeXt50_32x4d-dual          | 11.192G| 23.041M | 0.8696   | 0.8580    |
| +AHCR                         | 12.808G| 34.354M | 0.8721   | 0.8684    |
| +Ours                         | 14.604G| 47.774M | **0.8754**| **0.8761**|
| RegNet_x_3.2gf-dual           | 8.414G | 14.318M | 0.8291   | 0.8240    |
| +AHCR                         | 8.717G | 16.865M | 0.8352   | 0.8286    |
| +Ours                         | 9.226G | 20.089M | **0.8389**| **0.8430**|
| ConvNeXt_Tiny-dual            | 11.810G| 30.194M | 0.8823   | 0.8848    |
| +AHCR                         | 11.887G| 29.438M | 0.8971   | 0.8933    |
| +Ours                         | 12.435G| 31.381M | **0.9151**| **0.9098**|

## Requirements
The following dependencies are required to run the code:

- Python 3.x
- PyTorch 2.5.1
- numpy
- opencv-python
- scikit-learn
- scipy
- einops
- thop
- torchmetrics
- PyWavelets
- timm
- pandas
- openpyxl
- transformers>=4.5.0
- dill

You can install them using the following:

```bash
pip install -r requirements.txt
```
## Usage

### 1. Dataset Preparation
First, download and extract the dataset from the following link:

- [DvXray Dataset](https://github.com/Mbwslib/DvXray)

After downloading, extract the dataset to the `data/` directory in the root of the project.

### 2. Dataset Splitting
To split the dataset into training, validation, and test sets, use the `split_dataset.py` script. This script will automatically divide the dataset with a 7:2:1 ratio:

```
python split_dataset.py
```

### 3. Training the Model
Before training, ensure all dependencies are installed, and the necessary parameters are configured. You can start training by running the `train.py` script. If you are using multi-GPU training, enable `DataParallel` for model parallelism:

```
python train.py
```

### 4. Model Evaluation
After training, you can evaluate the model on the test set using the following command:

```
python train.py --eval -r <checkpoint>
```

### 5. Configuration File
You can specify a configuration file for training or evaluation by using the `--config` parameter. This allows you to easily adjust hyperparameters and model settings for different experiments.

## Citation
If you use this code or the method presented in the paper, please cite the following:

TODO: Add citation information here.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
