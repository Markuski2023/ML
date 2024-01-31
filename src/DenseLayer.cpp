#include "../include/Matrix.h"
#include "../include/Layers/DenseLayer.h"


// Implementation of a Dense Layer for neural networks using double as the data type
DenseLayer::DenseLayer(unsigned inputSize, unsigned outputSize)
        : weights(inputSize, outputSize), biases(1, outputSize) {
    // Constructor initializes a dense layer with given input and output sizes.
}

DenseLayer::DenseLayer() {
    // Default constructor
}

// Forward pass computation
Matrix<double> DenseLayer::forward(Matrix<double>A& input)  {
    this->input = input;  // Store the input matrix for use in backpropagation
    return (input.dotTiling(weights)) + biases;  // Compute output = (input * weights) + biases
}

// Backward pass computation
Matrix<double> DenseLayer::backward(Matrix<double>& output) {
    // Given the error of the output (derivative of loss w.r.t output), compute backpropagation step

    // Calculate gradient of weights as input^T * outputError
    Matrix<double> weightsGradient = input.transpose().dotTiling(output);

    // Calculate error w.r.t the input of this layer (needed for previous layer in the network)
    Matrix<double> inputError = output.dotTiling(weights.transpose());

    // Store gradients for weights and biases for later use in updateWeights
    weightsError = weightsGradient;
    biasesError = output.sum(0);  // Sum across columns to get biases error

    return inputError;  // Return the calculated input error
}

// Update Weights using an optimizer
void DenseLayer::updateWeights(Optimizer<double>& optimizer, double learningRate) {
    // Update weights and biases using the optimizer, based on stored gradients and learning rate
    optimizer.update(weights, biases, weightsError, biasesError, learningRate);
}

// Getters for weights and biases
Matrix<double>& DenseLayer::getWeights() {
    return weights;  // Return reference to weights matrix
}

Matrix<double>& DenseLayer::getBiases() {
    return biases;  // Return reference to biases matrix
}

// Setters for weights and biases
void DenseLayer::setWeights(Matrix<double>& newWeights) {
    // Set new weights, ensure size compatibility
    if (this->weights.get_rows() != newWeights.get_rows() || this->weights.get_cols() != newWeights.get_cols()) {
        this->weights = Matrix<double>(newWeights.get_rows(), newWeights.get_cols());
    }
    this->weights = newWeights;
}

void DenseLayer::setBiases(Matrix<double>& newBiases) {
    // Set new biases, ensure size compatibility
    if (this->biases.get_rows() != newBiases.get_rows() || this->biases.get_cols() != newBiases.get_cols()) {
        this->biases = Matrix<double>(newBiases.get_rows(), newBiases.get_cols());
    }
    this->biases = newBiases;
}
