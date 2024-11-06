# PGD Visualisation - Graph Generation

This repository contains code for generating visualizations of Projected Gradient Descent (PGD) attacks. The primary task here is to merge data from the `PCAClassAnalysis` file and the `PlotCosineSimilarity` file to demonstrate how attacks work.

## Example File

The `ExampleAttackTraining.py` file included in this repository is working code sourced from another repository. It serves as a demonstration of how PGD attacks are implemented and visualized. The example showcases the full training process but does not include the visualization of the attacks. Feel free to borrow the code for generating the attacks, or use it as a reference for your own implementation.

## Files

- **PCAClassAnalysis**: This file contains code for performing Principal Component Analysis (PCA) on the dataset to analyze class separability.
- **PlotCosineSimilarity**: This file includes code for plotting cosine similarity between different data points to visualize the effect of PGD attacks.

## Task

The main task is to merge the data from the `PCAClassAnalysis` file and the `PlotCosineSimilarity` file. This involves:

1. Extracting relevant plotted points from PCA 
2. Use the plotted points in the visualisation of loss.
3. See if you can use actual values for the visualisation of the loss
4. BONUS tasks - Can you get the radius of plotted points to vary based on attack direction and magnitude of the perturbation?


## Usage

To run the example and visualize the PGD attacks, follow these steps:

1. Ensure you have all the necessary dependencies installed.
2. Analyze the output to understand how PGD attacks affect the data.

## Acknowledgements

The example code is sourced from another repository and adapted for demonstration purposes in this project.
