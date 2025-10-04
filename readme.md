# Project 3 – Autoencoder / VAE: Dimensionality Reduction and Data Generation

Project 3 focuses on autoencoders and variational autoencoders, which are used for dimensionality reduction and data generation. The goal of the project is to build a classical autoencoder for image compression and reconstruction, as well as a variational autoencoder for generating new examples. The data used in the project comes from the MNIST and Fashion-MNIST datasets, which contain images of digits and clothing items, respectively, with a resolution of 28 by 28 pixels. Both datasets are available in the Keras and TorchVision libraries.

The project involves implementing a classical autoencoder and evaluating the quality of its reconstructions. Next, dimensionality reduction and visualization of the latent space should be performed using PCA or t-SNE. The following step is to implement a variational autoencoder and generate new images. The final step is to compare the classical autoencoder and the variational autoencoder in terms of reconstruction quality and generative capabilities.

Various datasets can be used in the project. The primary datasets are MNIST and Fashion-MNIST. Alternatively, CIFAR-10 or CIFAR-100 datasets containing color images, the CelebA dataset containing celebrity face images, or medical datasets such as MedMNIST v2 or HAM10000 can be used. The choice of dataset depends on the experimental goal, and different datasets allow testing the models under diverse conditions.

# Report 1 – Data Preparation

## Objective
To show how the dataset was prepared for training the Autoencoder (AE) and Variational Autoencoder (VAE).

## Report Content

- **Dataset Description**  
  Datasets: MNIST / Fashion-MNIST  
  Number of classes: 10  
  Number of images: dataset-dependent (e.g., 60,000 training images, 10,000 test images)

- **Data Normalization**  
  Data should be normalized to the range 0–1.

- **Data Split Scheme**  
  Data should be split into:
    - Train
    - Validation
    - Test

- **Sample Images from the Dataset**  
  A few rows of digit or clothing images for visual presentation of the data.
