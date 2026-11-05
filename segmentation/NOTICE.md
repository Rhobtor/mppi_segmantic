# Notice and Attribution

This repository contains research code for RGB-T semantic segmentation
in search-and-rescue (SAR) environments.

## Upstream CMX implementation

Part of this repository is derived from the official implementation of:

CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers

Upstream repository:

huaaaliu/RGBX_Semantic_Segmentation

Original CMX implementation:

Copyright (c) 2022 Huayao Liu

The upstream code is distributed under the MIT License. The original
license text is preserved in the root `LICENSE` file of this repository.

The CMX paper should be cited as:

Zhang, J., Liu, H., Yang, K., Hu, X., Liu, R., and Stiefelhagen, R.
"CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with
Transformers."
IEEE Transactions on Intelligent Transportation Systems, 2023.

## Research modifications

The `V2_0` and `V2_2` directories contain research modifications developed
for RGB-T semantic segmentation experiments in SAR environments.

The repository intentionally preserves CMX, V2.0, and V2.2 as separate
implementations to make architectural comparisons transparent and to
avoid obscuring experimental differences through refactoring.

## Dataset and model weights

The RGB-T dataset, pretrained backbone, and trained model checkpoints are
distributed separately from the source code.

Their presence is not required for cloning the source repository, but they
are required for reproducing training or evaluation.

See:

- `dataset/README.md`
- `pretrained/README.md`
- `checkpoints/README.md`

for the expected directory structure and integrity information.
