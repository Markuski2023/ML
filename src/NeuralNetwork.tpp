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
Matrix<T> NeuralNetwork<T>::backward(Matrix<T> &networkOutput, Matrix<T> &expectedOutput) {
    // Calculate the initial error gradient based on the network's output and the expected output.
    // This is the derivative of the loss function with respect to the network's output.
    Matrix<T> errorGradient = this->error(networkOutput, expectedOutput);

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

// Function to save the current parameters of a neural network into a JSON file.
// It is a template function, allowing it to handle different data types for the network parameters.
template <typename T>
void NeuralNetwork<T>::save(const std::string& filename) {

    // Create a new JSON object to store the network's parameters.
    nlohmann::json json;

    // Iterate through each layer in the network's 'layers' vector.
    for (size_t layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
        // Create a JSON object to store the current layer's information.
        nlohmann::json layerJson;

        // Retrieve the type of the current layer using Run Time Type Information (RTTI).
        // 'typeid' provides the type information and 'name()' returns its name as a char array.
        const char* mangledName = typeid(*layers[layerIndex]).name();
        int status;
        // Demangle the type name to make it human-readable.
        // 'abi::__cxa_demangle' demangles the name, and the result is wrapped in a unique_ptr for automatic memory management.
        std::unique_ptr<char, void(*)(void*)> demangledName(
                abi::__cxa_demangle(mangledName, NULL, NULL, &status),
                std::free
        );

        // Store the layer type in 'layerJson'.
        // If the demangling was successful (status == 0), use the demangled name; otherwise, use the mangled name.
        layerJson["type"] = (status == 0) ? demangledName.get() : mangledName;

        // Check if the current layer is of type 'DenseLayer'.
        auto denseLayer = std::dynamic_pointer_cast<DenseLayer<T>>(layers[layerIndex]);
        if (denseLayer) {
            // If it is a 'DenseLayer', extract and store its weights and biases.
            // 'matrixToJson' converts the matrix data into a JSON format.
            layerJson["weights"] = matrixToJson(denseLayer->getWeights());
            layerJson["biases"] = matrixToJson(denseLayer->getBiases());
        }

        // Add the JSON representation of the current layer to the main 'json' object.
        json["layers"].push_back(layerJson);
    }

    // Writing the JSON object to a file.
    std::ofstream file(filename);
    // Check if the file opened successfully.
    if (!file.is_open()) {
        // If the file fails to open, throw an exception.
        throw std::runtime_error("Unable to open file for saving weights and biases.");
    }
    // Write the JSON object to the file with an indentation of 4 spaces for readability.
    file << json.dump(4);
    // Close the file stream.
    file.close();
}

// Function template to convert a matrix into a JSON object.
// The template allows this function to work with matrices of different data types.
template <typename T>
nlohmann::json NeuralNetwork<T>::matrixToJson(Matrix<T>& matrix) {

    // Initialize an empty JSON object.
    // This object will eventually hold the matrix data in a structured format.
    nlohmann::json json;

    // Loop through each row of the matrix.
    for (size_t i = 0; i < matrix.get_rows(); ++i) {

        // Initialize a vector to represent a single row.
        // This vector will temporarily store the elements of the current row.
        std::vector<T> row;

        // Loop through each column of the current row.
        for (size_t j = 0; j < matrix.get_cols(); ++j) {

            // Add the current element to the row vector.
            // matrix(i, j) accesses the element at the ith row and jth column of the matrix.
            row.push_back(matrix(i, j));
        }

        // Add the completed row vector to the JSON object.
        // Each row vector becomes an array within the JSON array, representing a row in the matrix.
        json.push_back(row);
    }

    // Return the JSON object representing the matrix.
    // The JSON object is a 2D array mirroring the structure of the matrix.
    return json;
}

// Function to load the neural network parameters from a JSON file.
// The function is a part of the NeuralNetwork template class, which allows it to handle different data types.
template <typename T>
void NeuralNetwork<T>::load(const std::string& filename) {
    // Open the file for reading.
    std::ifstream file(filename);
    // Check if the file was successfully opened.
    if (!file.is_open()) {
        // If the file is not open, throw an exception.
        throw std::runtime_error("Unable to open file.");
    }

    // Create a JSON object.
    nlohmann::json json;
    // Read the contents of the file into the JSON object.
    file >> json;

    // Close the file.
    file.close();

    // Clear any existing layers in the network.
    layers.clear();

    // Loop through each layer in the JSON object.
    for (auto& layerJson : json["layers"]) {
        // Extract the layer type from the JSON object.
        std::string layerType = layerJson["type"];

        // Check if the layer type is 'DenseLayer<double>'.
        if (layerType == "DenseLayer<double>") {
            // Create a shared pointer to a new DenseLayer.
            auto layer = std::make_shared<DenseLayer<T>>();

            // Set the weights and biases for the layer.
            // Convert JSON arrays to matrices using the jsonToMatrix function.
            Matrix<T> weights = jsonToMatrix(layerJson["weights"]);
            Matrix<T> biases = jsonToMatrix(layerJson["biases"]);
            layer->setWeights(weights);
            layer->setBiases(biases);

            // Add the configured layer to the network.
            layers.push_back(layer);
        } else if (layerType == "ReLU") {
            // Create and add a ReLU layer if the layer type is 'ReLU'.
            auto layer = std::make_shared<ReLU>();
            layers.push_back(layer);
        }
        // Additional else-if statements can be added here for other layer types.
    }
}

// Function to convert a JSON array into a matrix.
// This function is useful for converting stored weights and biases back into their matrix form.
template <typename T>
Matrix<T> NeuralNetwork<T>::jsonToMatrix(const nlohmann::json& json) {
    // Determine the number of rows and columns from the JSON array.
    size_t rows = json.size();
    size_t cols = json.at(0).size();

    // Initialize a new matrix with the determined dimensions.
    Matrix<T> matrix(rows, cols);

    // Parallel for loop to populate the matrix with values from the JSON array.
    // '#pragma omp parallel for' is used for parallel processing to improve performance.
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // Assign each value from the JSON array to the corresponding element in the matrix.
            matrix(i, j) = json[i][j];
        }
    }
    // Return the populated matrix.
    return matrix;
}