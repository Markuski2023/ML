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
Matrix<double> DenseLayer::forwardPropagate(Matrix<double>& input)  {
    this->input = input;  // Store the input matrix for use in backpropagation
    Matrix<double> output = input.dotTiling(weights);

    // Manually add biases to each row of the output matrix
    for (size_t i = 0; i < output.get_rows(); ++i) {
        for (size_t j = 0; j < output.get_cols(); ++j) {
            output(i, j) += biases(0, j);
        }
    }
    return output;
}

// Backward pass computation for batched data
Matrix<double> DenseLayer::backwardPropagate(Matrix<double>& outputError) {
    // Calculate gradient of weights as sum of (input^T * outputError) for each data point in the batch
    // Then, average the gradient over the batch size
    Matrix<double> weightsGradient = input.transpose().dotTiling(outputError) / static_cast<double>(input.get_rows());

    // Calculate error w.r.t the input of this layer (needed for previous layer in the network)
    // This error is also averaged over the batch size
    Matrix<double> inputError = outputError.dotTiling(weights.transpose()) / static_cast<double>(outputError.get_rows());

    // Store gradients for weights and biases for later use in updateWeights
    // Averaging the biases gradient over the batch size
    biasesError = outputError.sum(0) / static_cast<double>(outputError.get_rows());

    // Store the averaged gradients
    weightsError = weightsGradient;
    biasesError = biasesError;

    return inputError;  // Return the calculated input error
}

// Update Weights using an optimizer
void DenseLayer::update(Optimizer<double>& optimizer, double learningRate) {
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
