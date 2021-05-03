# Multi-input Mixed Data Problem

Feed in mixed data from two inputs

input 1:
- numerical data (number of beds, baths, etc.)
- categorical data (zipcode, etc.)
  
input 2:
- image data (bath, bed, front, etc.)

input 1 > Multi-later Perceptron (MLP)

input 2 > Convolutional Neural Network (CNN)

concatenate outputs from MLP + CNN > Fully Connected (FC) + Linear Activation

..to predict home prices