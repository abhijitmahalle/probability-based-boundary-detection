[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Probability based edge detection
## Phase 1
This repository consits of a simple implementaion of [Probability of boundary detection](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf) algorithm and its comparsion with the classical approaches for edge detection like Canny and Sobel edge detectors that measures discontinuities in the image intensities to detect edges. The PB algorithm considers texture and color information along with intensity, making it a better performing algorithm. This algorithm predicts per pixel probability of the boundary detected. The original image and the output of implementation is shown below:

<img src="Phase1/data/BSDS500/Images/3.jpg" align="center" alt="Original" width="400"/> <img src="Phase1/results/PbLite/PbLite_3.png" align="center" alt="PBLite" width="400"/>

The algorithm of PBLite detection is shown below:

<img src="Phase1/results/hw0.png" align="center" alt="PBLite"/>

The main steps for implementing the same are:

### Step 1: Feature extraction using Filtering
The filter banks implemented for low-level feature extraction are Oriented Derivative if Gaussian Filters, Leung-Malik Filters (multi-scale) and Gabor Filter.

<img src="Phase1/results/DoG.png" align="center" alt="DoG" width="250"/> <img src="Phase1/results/LM.png" align="center" alt="PBLite" width="250"/> <img src="Phase1/results/Gabor.png" align="center" alt="PBLite" width="250"/>

### Step 2: Extracting texture, color and brightness using clustering
Filter banks can be used for extraction of texture properties but here all the three filter banks are combined which results into vector of filter responses. As filter response vectors are generated, they are clustered together using k-means clustering. For Texton Maps k = 64 is used; Color and Brightness Maps k= 16 is used.


<img src="Phase1/results/TextonMap/TextonMap_3.png" align="center" alt="DoG" width="250"/> <img src="Phase1/results/ColorMap/colormap_3.png" align="center" alt="PBLite" width="250"/> <img src="Phase1/results/BrightnessMap/BrightnessMap_3.png" align="center" alt="PBLite" width="250"/>

The gradient measurement is performed to know how much all features distribution is changing at a given pixel. For this purpose, half-disc masks are used.

<img src="Phase1/results/TextonGradient/Tg_3.png" align="center" alt="PBLite" width="250"/> <img src="Phase1/results/ColorGradient/Cg_3.png" align="center" alt="PBLite" width="250"/> <img src="Phase1/results/BrightnessGradient/Bg_3.png" align="center" alt="PBLite" width="250"/>

### Step 3: Pb-Score
The gradient maps which are generated are combined with classical edge detectors like Canny and Sobel baselines for weighted avaerage.

## Instructions to run the code:
```
python Wrapper.py
```
## File structure
    Phase1
    ├── Code
    |  ├── Wrapper.py
    ├── Data
    |  ├── BSDS500
    ├── results
    |  ├── BrightnessGradient
    |  ├── Brightness_map
    |  ├── ColorGradient
    |  ├── Color_map
    |  ├── PbLite
    |  ├── TextonGradient
    |  ├── TextonMap
    |  ├── Other filter outputs

This was implemented as part of [CMSC733](https://cmsc733.github.io/2022/hw/hw0/) and for detailed report refer [here](https://github.com/AbhijitMahalle/probability-based-boundary-detection/blob/master/Report.pdf).

## Phase 2
In this section, a basic neural network and its modified
version for classification on CIFAR10 dataset have been de-scribed. Later, a case study for ResNet, ResNext and DenseNet
architecture was conducted. Refer [report](https://github.com/AbhijitMahalle/probability-based-boundary-detection/blob/master/Report.pdf) for more details. 
