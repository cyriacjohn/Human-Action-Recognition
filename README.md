# Human Action Recognition using CNN + LSTM

## Overview

Human Action Recognition is a significant area of research in computer vision, where the goal is to automatically identify human actions in video sequences. This project leverages the power of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to achieve high accuracy in recognizing actions performed by individuals in video frames.

This repository contains a Jupyter Notebook that implements a model combining CNN and LSTM for recognizing human actions from video data.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Deep Learning Model**: Utilizes CNN for spatial feature extraction and LSTM for temporal sequence prediction.
- **High Accuracy**: Achieves significant performance improvements over traditional methods.
- **Modular Design**: Easily customizable and extensible for different datasets and action recognition tasks.

## Installation

To get started with the project, follow these steps:

1. **Clone the repository:**

2. **Install the required packages:**

Ensure you have Python installed, then install the necessary libraries:

Adjust parameters and configurations as needed to improve model performance.

## Dataset
This project uses the UCF101 dataset, a benchmark dataset for action recognition containing 13,320 videos categorized into 101 action classes. Ensure you download the dataset and set the path in the notebook.

## Model Architecture
The model consists of:

- Convolutional Neural Networks (CNN): For feature extraction from individual frames.
- Long Short-Term Memory (LSTM): For capturing temporal dynamics of the action sequence.
Model Diagram

## Results
The model's performance can be evaluated based on accuracy metrics provided in the notebook after training. You can visualize the results and fine-tune the model parameters for better performance.

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

-Fork the repository.
-Create a new branch (git checkout -b feature-branch).
-Make your changes and commit them (git commit -m 'Add new feature').
-Push to the branch (git push origin feature-branch).
-Create a new Pull Request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.




