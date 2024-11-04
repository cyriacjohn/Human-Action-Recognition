# Human Action Recognition using CNN + LSTM

![Human Action Recognition](https://miro.medium.com/v2/resize:fit:1200/format:webp/1*-kd-Nk1RauQ1GkJvlK0tWg.png)

## Overview

Human Action Recognition is a significant area of research in computer vision, where the goal is to automatically identify human actions in video sequences. This project leverages the power of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to achieve high accuracy in recognizing actions performed by individuals in video frames.

This repository contains a Jupyter Notebook that implements a model combining CNN and LSTM for recognizing human actions from video data.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
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

   ```bash
   git clone https://github.com/cyriacjohn/Human-Action-Recognition.git
   cd Human-Action-Recognition

Install the required packages:

Ensure you have Python installed, then install the necessary libraries:

bash
Copy code
pip install -r requirements.txt
Usage
Open the Jupyter Notebook:

bash
Copy code
jupyter notebook Copy_of_Human_Action_Recogntion_using_CNN_%2B_LSTM.ipynb
Run the cells in the notebook to train the model on the provided dataset.

Adjust parameters and configurations as needed to improve model performance.

Dataset
This project uses the UCF101 dataset, a benchmark dataset for action recognition containing 13,320 videos categorized into 101 action classes. Ensure you download the dataset and set the path in the notebook.

Model Architecture
The model consists of:

Convolutional Neural Networks (CNN): For feature extraction from individual frames.
Long Short-Term Memory (LSTM): For capturing temporal dynamics of the action sequence.
Model Diagram

Results
The model's performance can be evaluated based on accuracy metrics provided in the notebook after training. You can visualize the results and fine-tune the model parameters for better performance.

Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit them (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Create a new Pull Request.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Thank you for checking out this project! For any inquiries or issues, feel free to reach out or open an issue on the repository.


### Tips for Customizing the README

- **Images**: Update the URLs for any images you want to include in the README. You can add screenshots or results to make it more visually appealing.
- **Personal Touch**: Feel free to add your own insights, experiences, or specific implementation details that you think would benefit users.
- **Links**: Ensure all links (e.g., dataset links) are functional and point to the correct resources.

This README structure provides a comprehensive overview of your project, making it easier for others to understand and contribute. Let me know if you want to add or modify any sections!
