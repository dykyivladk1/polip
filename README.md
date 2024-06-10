
# PyTorch Library for ML Projects

![Polip](images/polip.png)

## Introduction
This is a comprehensive library designed to facilitate various machine learning projects using PyTorch. It provides essential functionalities such as custom layers, dataset handling, and utility functions for model training and visualization.

## Features
- Custom layers for pixel normalization and upsampling/downsampling.
- Convenient data transformation and augmentation functions.
- Custom dataset class for handling image datasets.
- Utility functions for model initialization, visualization, and more.

## Installation
This tool requires Python. Use this command to install the library:
\`\`\`bash
pip install polip
\`\`\`

## Required Libraries for Visualization
Make sure to install the following required libraries:
\`\`\`bash
pip install matplotlib os torch PIL numpy torchvision
\`\`\`

## Usage

### Custom Layers
The library includes custom layers like \`PixelNormLayer\`, \`UpSample\`, and \`DownSample\`. Here's an example of how to use them:

\`\`\`python
from polip.cb import PixelNormLayer, UpSample, DownSample
\`\`\`

### Custom Image Dataset
You can use the \`CustomImageDataset\` class to handle image datasets:

\`\`\`python
from polip import CustomImageDataset, get_rgb_transform
\`\`\`

