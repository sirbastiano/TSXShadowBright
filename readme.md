
# XShadowBright Dataset Segmentation

This repository contains code for segmenting the bright “shadow” from SAR imagery of moving ships, specifically targeting cross-polarized (HV/VH) images. The segmentation task is crucial for estimating ship velocity and analyzing ship-induced disturbances on the ocean surface.

The code is structured with the primary logic in `main.ipynb` and auxiliary functions, including loss definitions, in separate files for modularity.

## Dataset

The **XShadowBright** dataset contains 1,100 samples, split into:

- **Train:** 50%
- **Validation:** 20%
- **Test:** 30%

Each sample consists of cross-polarized SAR images from TerraSAR-X, along with corresponding masks delineating the bright “shadow” region. These samples have been augmented with spatial and noise-based transformations to increase dataset diversity and robustness.

You can download the dataset from [Zenodo](https://zenodo.org/records/14844141).
