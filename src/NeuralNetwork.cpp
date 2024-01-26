#include "../include/NeuralNetwork.h"
#include "../include/Layers/DenseLayer.h"
#include "../include/Layers/ReLU.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <typeinfo>
#include <cxxabi.h>  // Include for demangling
#include <memory>    // Include for unique_ptr
#include "../json.hpp"
#include <omp.h>

Matrix<double> NeuralNetwork::backward(Matrix<double> &networkOutput, Matrix<double> &expectedOutput) {
    // Calculate the initial error gradient based on the network's output and the expected output.
    // This is the derivative of the loss function with respect to the network's output.
    Matrix<double> errorGradient = error->calculateError(networkOutput, expectedOutput);

    // Iterate over the layers in reverse order (from output layer to input layer).
    for (auto reverseLayerIterator = layers.rbegin(); reverseLayerIterator != layers.rend(); ++reverseLayerIterator) {
        // Call the backward method of each layer, passing the current error gradient.
        // Each layer updates the error gradient based on its own parameters and operations.
        // This updated error gradient is then passed to the preceding layer in the next iteration.
        errorGradient = (*reverseLayerIterator)->backward(errorGradient);
    }

    // Return the final error gradient after it has propagated through all layers.
    return errorGradient;
}

Matrix<double> NeuralNetwork::forward(Matrix<double> &input) {
    // Initialize the output of the network as the input. This output will be modified
    // as it passes through each layer.
    Matrix<double> output = input;

    // Iterate over each layer in the neural network.
    // The order of iteration is from the input layer to the output layer.
    for (auto& layer : layers) {
        // Call the forward method of each layer, passing the current output.
        // Each layer processes the output from the previous layer and produces
        // a new output.
        output = layer->forward(output);
    }
    // After passing through all layers, return the final output.
    return output;
}

void NeuralNetwork::addLayer(std::shared_ptr<Layer> layer) {
    // The method takes a shared pointer to a Layer object as an input.
    // This allows for polymorphism, as different types of layers (like DenseLayer,
    // ActivationLayer, etc.) can be added to the neural network.

    // Add the given layer to the end of the 'layers' vector.
    layers.push_back(layer);
}

void NeuralNetwork::updateNetworkWeights() {
    for (auto& layer : layers) {
        // Dynamic cast to check if the layer is a DenseLayer
        auto denseLayer = std::dynamic_pointer_cast<DenseLayer>(layer);
        if (denseLayer) {
            // Update weights using the optimizer and learning rate
            denseLayer->updateWeights(*optimizer, learningRate);
        }
    }
}

std::pair<std::vector<Matrix<double>>, std::vector<Matrix<double>>> NeuralNetwork::getWeights() {
    std::vector<Matrix<double>> weights;
    std::vector<Matrix<double>> biases;

    for (auto& layer : layers) {
        // Dynamic cast to check if the layer is a DenseLayer
        auto denseLayer = std::dynamic_pointer_cast<DenseLayer>(layer);
        if (denseLayer) {
            // Update weights using the optimizer and learning rate
            weights.push_back(denseLayer->getWeights());
            biases.push_back(denseLayer->getBiases());
        }
    }
    return std::pair(weights, biases);
}

double calculateTotalError(Matrix<double>& errorGradient) {
    double totalError = 0.0;
    for (size_t i = 0; i < errorGradient.get_rows(); ++i) {
        for (size_t j = 0; j < errorGradient.get_cols(); ++j) {
            totalError += errorGradient(i, j);
        }
    }
    return totalError;
}

double NeuralNetwork::getLastErrorValue() {
    return lastErrorValue;
}

void NeuralNetwork::setLastErrorValue(double errorValue) {
    lastErrorValue = errorValue;
}

std::string NeuralNetwork::returnErrorName() {

}

std::string NeuralNetwork::returnOptimizerName() {

}

// Function to save the current parameters of a neural network into a JSON file.
// It is a template function, allowing it to handle different data types for the network parameters.
void NeuralNetwork::save(const std::string& filename) {

}

// Function template to convert a matrix into a JSON object.
// The template allows this function to work with matrices of different data types.
nlohmann::json NeuralNetwork::matrixToJson(Matrix<double>& matrix) {

}

// Function to load the neural network parameters from a JSON file.
// The function is a part of the NeuralNetwork template class, which allows it to handle different data types.
void NeuralNetwork::load(const std::string& filename) {

}

// Function to convert a JSON array into a matrix.
// This function is useful for converting stored weights and biases back into their matrix form.
Matrix<double> NeuralNetwork::jsonToMatrix(const nlohmann::json& json) {

}