#include "../include/Matrix.h"
#include "../include/Layers/DenseLayer.h"
#include <memory>
#include <vector>

// Constructor
template <typename T>
DenseLayer<T>::DenseLayer(unsigned inputSize, unsigned outputSize)
        : weights(inputSize, outputSize), biases(1, outputSize) {}

template <typename T>
DenseLayer<T>::DenseLayer() {
    // Implementation of default constructor
    // Initialize weights, biases, and other members as needed
}

// Forward
template <typename T>
Matrix<T> DenseLayer<T>::forward(Matrix<T>& input)  {
    this->input = input;
    return (input.dotTiling(weights)) + biases;
}


// Backward
template <typename T>
Matrix<T> DenseLayer<T>::backward(Matrix<T>& outputError) {
    // Calculate gradient with respect to weights
    Matrix<T> weightsGradient = input.transpose().dotTiling(outputError);

    // Calculate the error with respect to the input of this layer
    Matrix<T> inputError = outputError.dotTiling(weights.transpose());

    // Save gradients for weights and biases for later use in updateWeights
    weightsError = weightsGradient;
    biasesError = outputError.sum(0);

    return inputError;
}

// Update Weights
template <typename T>
void DenseLayer<T>::updateWeights(Optimizer<T>& optimizer, double learningRate) {
    optimizer.update(weights, biases, weightsError, biasesError, learningRate);
}

template <typename T>
Matrix<T>& DenseLayer<T>::getWeights() {
    return weights;
}

template <typename T>
Matrix<T>& DenseLayer<T>::getBiases() {
    return biases;
}

template <typename T>
void DenseLayer<T>::setWeights(Matrix<T>& newWeights) {
    if (this->weights.get_rows() != newWeights.get_rows() || this->weights.get_cols() != newWeights.get_cols()) {
        this->weights = Matrix<T>(newWeights.get_rows(), newWeights.get_cols());
    }
    this->weights = newWeights;
}

template <typename T>
void DenseLayer<T>::setBiases(Matrix<T>& newBiases) {
    if (this->biases.get_rows() != newBiases.get_rows() || this->biases.get_cols() != newBiases.get_cols()) {
        this->biases = Matrix<T>(newBiases.get_rows(), newBiases.get_cols());
    }
    this->biases = newBiases;
}
