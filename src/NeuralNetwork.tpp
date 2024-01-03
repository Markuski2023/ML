#include "../include/NeuralNetwork.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <typeinfo>
#include <cxxabi.h>  // Include for demangling
#include <memory>    // Include for unique_ptr
#include "../json.hpp"
#include <omp.h>

template <typename T>
// Method to pass error gradient from output layer to input layer -> uses a reverse iterator
Matrix<T> NeuralNetwork<T>::backward(Matrix<T> &output, Matrix<T> &target) {
    // Calculate the initial error gradient based on the output of the network and the target.
    // This is the derivative of the loss function with respect to the network's output.
    // Here, a simple Mean Squared Error (MSE) derivative is used: 2 * (output - target).
    Matrix<T> errorGradient = (output - target) * 2;
    // Iterate over the layers in reverse order (from output layer to input layer).
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        // Call the backward method of each layer, passing the current error gradient.
        // Each layer updates the error gradient based on its own parameters and operations.
        // This updated error gradient is then passed to the preceding layer in the next iteration.
        errorGradient = (*it)->backward(errorGradient);
    }
    // Return the final error gradient after it has propagated through all layers.
    // This can be used for further calculations if needed.
    return errorGradient;
}

template <typename T>
Matrix<T> NeuralNetwork<T>::forward(Matrix<T> &input) {
    // Initialize the output of the network as the input. This output will be modified
    // as it passes through each layer.
    Matrix<T> output = input;

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

template <typename T>
void NeuralNetwork<T>::addLayer(std::shared_ptr<Layer> layer) {
    // The method takes a shared pointer to a Layer object as an input.
    // This allows for polymorphism, as different types of layers (like DenseLayer,
    // ActivationLayer, etc.) can be added to the neural network.

    // Add the given layer to the end of the 'layers' vector.
    // This vector maintains the sequence of layers in the neural network.
    layers.push_back(layer);
}

template <typename T>
void NeuralNetwork<T>::updateNetworkWeights() {
    for (auto& layer : layers) {
        // Dynamic cast to check if the layer is a DenseLayer
        auto denseLayer = std::dynamic_pointer_cast<DenseLayer<T>>(layer);
        if (denseLayer) {
            // Update weights using the optimizer and learning rate
            denseLayer->updateWeights(*optimizer, learningRate);
        }
    }
}

template <typename T>
std::pair<std::vector<Matrix<T>>, std::vector<Matrix<T>>> NeuralNetwork<T>::getWeights() {
    std::vector<Matrix<T>> weights;
    std::vector<Matrix<T>> biases;

    for (auto& layer : layers) {
        // Dynamic cast to check if the layer is a DenseLayer
        auto denseLayer = std::dynamic_pointer_cast<DenseLayer<T>>(layer);
        if (denseLayer) {
            // Update weights using the optimizer and learning rate
            weights.push_back(denseLayer->getWeights());
            biases.push_back(denseLayer->getBiases());
        }
    }
    return std::pair(weights, biases);
}

template <typename T>
void NeuralNetwork<T>::save(const std::string& filename) {
    nlohmann::json json;

    for (size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
        nlohmann::json layerJson;

        // Get the layer type using RTTI (Run Time Type Information)
        const char* mangledName = typeid(*layers[layerIndex]).name();
        int status;
        std::unique_ptr<char, void(*)(void*)> demangledName(
                abi::__cxa_demangle(mangledName, NULL, NULL, &status),
                std::free
        );

        // Add layer type to JSON
        layerJson["type"] = (status == 0) ? demangledName.get() : mangledName;

        // Check if the layer is a DenseLayer
        auto denseLayer = std::dynamic_pointer_cast<DenseLayer<T>>(layers[layerIndex]);
        if (denseLayer) {
            layerJson["weights"] = matrixToJson(denseLayer->getWeights());
            layerJson["biases"] = matrixToJson(denseLayer->getBiases());
        }

        json["layers"].push_back(layerJson);
    }

    // Write JSON to file
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for saving weights and biases.");
    }
    file << json.dump(4); // 4 is the indentation level for pretty printing
    file.close();
}

template <typename T>
nlohmann::json NeuralNetwork<T>::matrixToJson(Matrix<T>& matrix) {
    nlohmann::json json;
    for (size_t i = 0; i < matrix.get_rows(); ++i) {
        std::vector<T> row;
        for (size_t j = 0; j < matrix.get_cols(); ++j) {
            row.push_back(matrix(i, j));
        }
        json.push_back(row);
    }
    return json;
}

template <typename T>
void NeuralNetwork<T>::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file.");
    }

    nlohmann::json json;
    file >> json;

    file.close();

    // Clear existing layers (if any)
    layers.clear();

    for (auto& layerJson : json["layers"]) {
        std::string layerType = layerJson["type"];

        if (layerType == "DenseLayer<double>") {
            auto layer = std::make_shared<DenseLayer<T>>();

            // Set weights and biases
            Matrix<T> weights = jsonToMatrix(layerJson["weights"]);
            Matrix<T> biases = jsonToMatrix(layerJson["biases"]);
            layer->setWeights(weights);
            layer->setBiases(biases);

            layers.push_back(layer);
        } else if (layerType == "ReLU") {
            // Create ReLU layer
            auto layer = std::make_shared<ReLU>();
            layers.push_back(layer);
        }
        // Add more else if statements for other layer types if needed
    }
}

template <typename T>
Matrix<T> NeuralNetwork<T>::jsonToMatrix(const nlohmann::json& json) {
    size_t rows = json.size();
    size_t cols = json.at(0).size();

    Matrix<T> matrix(rows, cols);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) = json[i][j];
        }
    }
    return matrix;
}
