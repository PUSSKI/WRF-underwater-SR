# WRF-Net: Region-Aware Frequency Modeling with Controlled Enhancement Feedback for Underwater Image Super-Resolution

This repository provides the official PyTorch implementation of **WRF-Net (Wavelet Region-Frequency Network)** for underwater image super-resolution.

WRF-Net addresses spatially heterogeneous underwater degradation by adaptively coordinating low- and high-frequency information according to local reconstruction demand, while controlled enhancement feedback introduces appearance restoration cues without dominating the super-resolution objective. The repository contains the model implementation, training and testing scripts, configuration files, and evaluation code. Experiments are conducted on **UFO-120** for underwater super-resolution and on **LSUI** and **UIEB** for underwater image enhancement evaluation.

---

## Environment

The code is implemented in **PyTorch** and is built on the NAFNet / BasicSR codebase.

```bash
conda create -n wrfnet python=3.8 -y
conda activate wrfnet

pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

Please install a PyTorch version compatible with your CUDA environment.

---

## Dataset Information

This project uses the following underwater image datasets:

- **UFO-120**: used for underwater image super-resolution training and evaluation at ×2, ×3, and ×4 scales.
- **LSUI**: used to train the separate underwater image enhancement configuration and for reference-based enhancement evaluation.
- **UIEB**: used for reference-based and no-reference underwater image enhancement evaluation.

These datasets are third-party datasets and are not redistributed in this repository. Users should obtain them from their original sources and comply with the corresponding terms of use.

## Dataset Preparation

### UFO-120 [Data](http://irvlab.cs.umn.edu/resources/ufo-120-dataset)

Prepare paired LR and HR images following the paths specified in the YAML configuration files.

Example structure:

```text
datasets/
└── UFO-120/
    ├── train_val/
    │   ├── hr/
    │   └── lrd/
    └── TEST/
        ├── hr/
        └── lrd/
```

Modify the dataset paths in the corresponding YAML files before training or testing.

### LSUI / UIEB [Data](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement) / [Data](https://li-chongyi.github.io/proj_benchmark.html)

LSUI is used for the separate underwater image enhancement experiments.  
UIEB is used for reference-based and no-reference enhancement evaluation.

---

## Training

Training is controlled by YAML configuration files.

For example, the ×2 UFO-120 model can be trained with:

```bash
python basicsr/train.py -opt options/train/UWNAFWaveCCSR/UWNAFWaveCCSR_x2.yml
```

Before training, check the dataset paths in the YAML file:

```yaml
datasets:
  train:
    dataroot_gt: /path/to/HR
    dataroot_lq: /path/to/LR

  val:
    dataroot_gt: /path/to/HR
    dataroot_lq: /path/to/LR
```

Training logs, checkpoints, and validation results are saved under the experiment directories managed by BasicSR.

---

## Testing

Set the pretrained model path and dataset paths in the test YAML file:

```yaml
path:
  pretrain_network_g: /path/to/net_g_best.pth
```

Then run:

```bash
python basicsr/test.py -opt options/test/UWNAFWaveCCSR/UWNAFWaveCCSR_x2.yml
```

The reconstructed images will be saved automatically according to the BasicSR result-directory settings.

---

## Repository Structure

```text
WRF-underwater-SR/
├── basicsr/
│   ├── archs/          # Network architectures
│   ├── data/           # Dataset and dataloader definitions
│   ├── losses/         # Loss functions
│   ├── models/         # Training/testing model wrappers
│   ├── train.py
│   └── test.py
├── options/
│   ├── train/          # Training configuration files
│   └── test/           # Testing configuration files
├── experiments/        # Checkpoints and training logs
├── results/            # Testing results
├── requirements.txt
└── README.md
```

---

## Code Information

The code is implemented in PyTorch using the BasicSR framework. WRF-Net is implemented as the `UWNAFWaveCCSRLocal` generator, and the training and testing procedures are controlled through YAML configuration files.

## Main Configuration

The released UFO-120 ×2 configuration uses:

```yaml
network_g:
  type: UWNAFWaveCCSRLocal
  up_scale: 2
  use_enhance_head: true
  use_enhance_feedback: true
  use_wavelet_attention: true
  wavelet_attention_blocks: 2
  use_region_fusion: true
  use_local_dense_trunk: true
```

The exact network settings, loss weights, residual scaling factors, and training hyperparameters are provided in the released YAML configuration files.

---

## License

The source code in this repository is provided for academic research purposes. Third-party datasets are subject to their respective licenses and terms of use and are not covered by this repository's license.
