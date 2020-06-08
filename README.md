# Optical Digit Recognition (OCR) - Java using Neural Network and K-Nearest Neighbour
Optical Handwritten Digits Recognition from a subset of the UCI (http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of_Handwritten+Digits) dataset.


## Properties of Dataset
* 2810 Training rows and 2810 Test rows
* all digits are represented by 64 (8 x 8) pixels with values ranging between 0 to 16.

## Neural Network Model
- 64 Inputs
- 1 Hidden Layer and 10 outputs
- 10 outputs (determine the digit expected for digits 0, 1, 2, ..., 9)

## Improved K-Nearest Neighbour Model
- looks for the closest matching digit based in the distance of the 32 inputs

## Results
- Accuracy of Neural Net: approx. 65 %
- Accuracy of K-Nearest Neighbout: approx. 98 %
- Algorithm is able to obtain optimal results.


# Take Aways
* Neural network can be optimized using deep learning to obtain higher accuracy results
* more efficient activation functions could be used to classify the digits
* K-NN is a simple, fast and reliable algorithm when it comes to classification problems

<b>Please refer to `OCR Rémi Wong.pdf` documentation to read more on the Handwritten Optical Digit Recognition problem</b>
