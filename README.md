# Music Genre Classification using CNN

Welcome to Music Genre Classification Project using Convolutional Neural Networks (CNN) to classify music tracks into genres. The aim of this project is to experiment on CNN to capture intricate patterns in the audio data in order to enable genre prediction.

## Overview

In this work, a CNN model is built and trained on the GTZAN dataset to classify songs into predefined genres using Mel-spectrograms or Mel-frequency cepstral coefficients (MFCCs) generated from audio files. This can be useful for music recommendations, organization, and more.

The training of the CNN model is meant to be used on a single CPU. To reduce data usage, features extraction and data transformations are applied on the fly.

## Installation

Clone this repository:
`
git clone https://github.com/QKTheoNguyen/Music_Classification_Project.git
`

Install the required packages:
`
pip install -r requirements.txt
`

## Usage

1. Download the Dataset: First Download the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
1. Prepare the Dataset: Ensure your dataset of audio files is organized by genre in the "data" folder
2. Train the Model: Execute the `main.py` script to train the CNN
3. Evaluate the Model: Use the evaluation script `evaluate.py` to test the model’s accuracy


