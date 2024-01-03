#include "../include/Matrix.h"
#include "../include/Layers/DenseLayer.h"
#include <memory>
#include <vector>

// Template implementation of a Dense Layer for neural networks
template <typename T>
DenseLayer<T>::DenseLayer(unsigned inputSize, unsigned outputSize)
        : weights(inputSize, outputSize), biases(1, outputSize) {
    // Constructor initializes a dense layer with given input and output sizes.
}

template <typename T>
DenseLayer<T>::DenseLayer() {
    // Default constructor
}

// Forward pass computation
template <typename T>
Matrix<T> DenseLayer<T>::forward(Matrix<T>& input)  {
    this->input = input;  // Store the input matrix for use in backpropagation
    return (input.dotTiling(weights)) + biases;  // Compute output = (input * weights) + biases
}


// Backward pass computation
template <typename T>
Matrix<T> DenseLayer<T>::backward(Matrix<T>& outputError) {
    // Given the error of the output (derivative of loss w.r.t output), compute backpropagation step

    // Calculate gradient of weights as input^T * outputError
    Matrix<T> weightsGradient = input.transpose().dotTiling(outputError);

    // Calculate error w.r.t the input of this layer (needed for previous layer in the network)
    Matrix<T> inputError = outputError.dotTiling(weights.transpose());

    // Store gradients for weights and biases for later use in updateWeights
    weightsError = weightsGradient;
    biasesError = outputError.sum(0);  // Sum across columns to get biases error

    return inputError;  // Return the calculated input error
}

// Update Weights using an optimizer
template <typename T>
void DenseLayer<T>::updateWeights(Optimizer<T>& optimizer, double learningRate) {
    // Update weights and biases using the optimizer, based on stored gradients and learning rate
    optimizer.update(weights, biases, weightsError, biasesError, learningRate);
}

// Getters for weights and biases
template <typename T>
Matrix<T>& DenseLayer<T>::getWeights() {
    return weights;  // Return reference to weights matrix
}

template <typename T>
Matrix<T>& DenseLayer<T>::getBiases() {
    return biases;  // Return reference to biases matrix
}

// Setters for weights and biases
template <typename T>
void DenseLayer<T>::setWeights(Matrix<T>& newWeights) {
    // Set new weights, ensure size compatibility
    if (this->weights.get_rows() != newWeights.get_rows() || this->weights.get_cols() != newWeights.get_cols()) {
        this->weights = Matrix<T>(newWeights.get_rows(), newWeights.get_cols());
    }
    this->weights = newWeights;
}

template <typename T>
void DenseLayer<T>::setBiases(Matrix<T>& newBiases) {
    // Set new biases, ensure size compatibility
    if (this->biases.get_rows() != newBiases.get_rows() || this->biases.get_cols() != newBiases.get_cols()) {
        this->biases = Matrix<T>(newBiases.get_rows(), newBiases.get_cols());
    }
    this->biases = newBiases;
}