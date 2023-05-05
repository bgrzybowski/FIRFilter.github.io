# FIRFilter.github.io
ECE 6530 DSP - Project Group 3 - Finite Impulse Response (FIR) Filters

 ## Project Description
 In this project MATLAB code is applied to illustrate:
- Deconvolution with 1-D Filters 
- Cascading of Two Systems
- FIR Filtering of Images
- Finding Edges: 1-D Filter Cascaded with Nonlinear Operators
- Bar Code Detection and Decoding

## Overview
This project was composed of three main exercises:

- Adding and echo to an audio signal and using an FIR filter to remove the echo and recover the original audio signal.

![image for audio signal processing](https://github.com/bgrzybowski/FIRFilter.github.io/blob/main/doc/AudioSignalEchoEquations.PNG)

- Applying a 1-D FIR filter to process a 2-D image by filtering the data matrix of the image first by horizontal rows, and then by vertical rows. Utilizing a cascaded filter to deconvolve a filtered image and recover the original image. 

![image for image filtering and deconvolution](https://github.com/bgrzybowski/FIRFilter.github.io/blob/main/doc/CascadingTwoSystemsDiagram.PNG)

- Utilization of a 1-D FIR filter to process a row of data from an image of a UPC symbol via edge detection. The bars in the UPC symbol are edges that represent a code of digits and can be decoded to a 12 digit sequence of numbers. 

![image for UPC decoding block diagram](https://github.com/bgrzybowski/FIRFilter.github.io/blob/main/doc/UPCdiagram.PNG)

## How to use this project
The MATLAB script file for this project is located in the src (source code) directory. The script uses several audio files, image files, and matlab data files. All the files required for the MATLAB script to run properly are stored in the dep directory (dependent files). The dependent files should be located in the same MATLAB working directory as the script when running the script file in MATLAB for best results. 

### Source Code: 
- ProjectGroup3.m

### Dependent Files: 
- decodeUPC.p
- echart.mat
- HP110v3.png
- labdat.mat
- labdat.wav
- labdat_eco_1.wav
- labdat_eco_2.wav
- labdat_original.wav
- matlab.mat
- OFFv3.png

## References

[1] J. F. Proakis, D. G. Manolakis, Digital Signal Processing: Principles, Algorithms, and Applications. Upper Saddle River, NJ, USA: Pearson Prentice Hall, 2007, pp. 660-664.

[2] McClellan, Schafer, and Yoder, Signal Processing First. Upper Saddle River, NJ, USA: Prentice Hall, Lab P-9: Sampling, Convolution, and FIR Filtering, pp. 1-11.

[3] McClellan, Schafer, and Yoder, Signal Processing First. Upper Saddle River, NJ, USA: Prentice Hall, Lab P-10: Edge Detection in Images: UPC Decoding, pp. 1-8.

# System Requirements

## Hardware
The script files were developed on Windows OS 7 or later, with Intel i9 or better processors. Older OS versions or slower processors may result in degraded performance. 

## Software
The script files for this project were developed in MATLAB R2021b and later versions. It is recommended that MATLAB R2021b or later versions be utilized for running the scripts. The scripts have not been tested in older versions of MATLAB.
