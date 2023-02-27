# Probability based edge detection
# Phase 1
My homework 0 submission for the course [CMSC733 Computer Processing of Pictorial Information](https://cmsc733.github.io/2021/hw/hw0/).
In this repository, a simple of version of PB boundary detection
algorithm has been implemented. The classical approaches
like Canny and Sobel edge detectors measures discontinuities
in the image intensities to detect edges. The PB algorithm
considers texture and color information along with intensity,
making it a better performing algorithm.
## Sample input image
<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/data/BSDS500/Images/10.jpg" width="300" height="300">

## Filter banks
Filter banks are a set of filters, that are applied over an
image to extract multiple features. In this homework, filter
banks have been used to extract the texture properties. The
following subsections will describe in brief the implementation
of DoG filters, Leung-Malik filters and Gabor filters.
### Oriented Derivative of Gaussian Filters      
<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/Filters/DoG.png" width="500" height="500">     

### Leung-Malik Filters: LM large and LM small             
<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/Filters/LM.png" width="500" height="500">      

### Gabor Filters       
<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/Filters/Gabor.png" width="500" height="500">        

## Texton Maps    
The filters described in the previous section are to detect
the texture properties in an image. Since each filter bank has
multiple filters and three such filter banks have been used,
the result is a vector of filter responses. This vector of filter
response associated with each pixel is encoding a texture
property. Based on this filter response vectors, pixels with
similar texture property were clustered together using K-mean
algorithm(K= 64). The output of K-mean clustering is a Texton
map(τ )

<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/Textron_map/TextonMap_10.jpg" width="300" height="300">      

## Brightness Maps    
The image was clustered based on the brightness value for
each pixel. The images were first converted to gray scale and
K-mean algorithm(K=16) was used to get brightness maps.    


<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/Brightness_map/BrightnessMap_10.jpg" width="300" height="300">     

## Color Maps
The image consists of three color channals(RGB), describ-
ing the color property at each pixel. The images have been
clustered using the RGB value using K-mean algorithm(K=16)
to obtain color maps.   

<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/Color_map/ColorMap_10.jpg" width="300" height="300">    

## Texture, Brightness and Color Gradients   
To obtain Tg,Bg,Cg, we need to compute differences of values across different shapes and sizes. This can be achieved very efficiently by the use of Half-disc masks.    

### Half disk masks
<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/Filters/HDMasks_12.png" width="300" height="300">     

### Texture Gradient
<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/T_g/tg_10.jpg" width="300" height="300">      

### Brightness Gradient
<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/B_g/bg_10.jpg" width="300" height="300">       

### Color Gradient
<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/C_g/cg_10.jpg" width="300" height="300">    

## Sobel and Canny baseline
The outputs from Sobel and Canny edge detector are
combined using weighted average method.   
### Sobel baseline    
<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/data/BSDS500/SobelBaseline/10.png" width="300" height="300">     

### Canny baseline    
<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/data/BSDS500/CannyBaseline/10.png" width="300" height="300">     

## Pb-lite output
In the final step, the features from baseline methods(Canny
and Sobel operator) were combined with the gradients of τ ,
β, ζ.    


<img src="https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Phase1/results/pb_lite_output/10.png" width="300" height="300"> 

# Running the code
# File structure
    .
    ├── Code
    |  ├── Wrapper.py
    ├── Data
    |  ├── BSDS500
    ├── Results
    |  ├── Filters
    |  ├── Testron_map
    |  ├── Brightness_map
    |  ├── Color_map
    |  ├── T_g
    |  ├── B_g
    |  ├── C_g
    |  ├── pb_lite_output
    

# How to run the code
- Change the location to      
```{root_directory}/Phase1 ```  

- Run the following command       
```python3 Code/Wrapper.py ```

# Phase 2
In this section, a basic neural network and its modified
version for classification on CIFAR10 dataset have been de-
scribed. Later, a case study for ResNet, ResNext and DenseNet
architecture was conducted. Refer [report](https://github.com/sakshikakde/probability-based-edge-detection/blob/main/Report.pdf) for more details. 
<!-- 
Phase 2:
1)train.py
following are the parameters that need to updated before running the code:
BasePath, CheckPointPath, NumEpochs, DivTrain, MiniBatchSize, LoadCheckPoint, LogsPath, Plotpath
2)test.py
following are the parameters that need to updated before running the code:
ModelPath, BasePath, TxtPath, Plotpath -->
