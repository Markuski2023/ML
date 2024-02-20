#include "../include/Matrix.h"
#include "../include/Layers/DenseLayer.h"

DenseLayer::DenseLayer(unsigned inputSize, unsigned outputSize)
        : weights(inputSize, outputSize), biases(1, outputSize) {}

DenseLayer::DenseLayer() {}

// Forward pass computation
Matrix<double> DenseLayer::forwardPropagate(Matrix<double>& input)  {
    this->input = input;  // Store the input matrix for use in backpropagation
    Matrix<double> output = input.dotTiling(weights);

    // Manually add biases to each row of the output matrix
    for (size_t i = 0; i < output.getRows(); ++i) {
        for (size_t j = 0; j < output.getCols(); ++j) {
            output(i, j) += biases(0, j);
        }
    }
    return output;
}

// Backward pass computation for batched data
Matrix<double> DenseLayer::backwardPropagate(Matrix<double>& currentLayerError) {
    // Averaging the values over the batch size
    Matrix<double> weightGradients = input.transpose().dotTiling(currentLayerError) / static_cast<double>(input.getRows());
    Matrix<double> biasGradients = currentLayerError.sum(0) / static_cast<double>(currentLayerError.getRows());

    // Store the averaged gradients
    storedWeightGradients = weightGradients;
    storedBiasGradients = biasGradients;

    Matrix<double> propagatedError = currentLayerError.dotTiling(input.transpose()) / static_cast<double>(currentLayerError.getRows());

    return propagatedError;  // Return the calculated input error
}

// Update Weights using an optimizer
void DenseLayer::update(Optimizer<double>& optimizer, double learningRate) {
    // Update weights and biases using the optimizer, based on stored gradients and learning rate
    optimizer.update(weights, biases, storedWeightGradients, storedBiasGradients, learningRate);
}

Matrix<double> & DenseLayer::getWeights() {
}

Matrix<double> & DenseLayer::getBiases() {
}
