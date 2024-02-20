#include "../include/NeuralNetwork.h"
#include "../include/Layers/DenseLayer.h"
#include <filesystem>
#include <string>
#include <typeinfo>
#include <cxxabi.h>
#include <memory>
#include "../json.hpp"

Matrix<double> NeuralNetwork::backward(Matrix<double> &networkOutput, Matrix<double> &expectedOutput) {
    Matrix<double> errorGradient = error->calculateError(networkOutput, expectedOutput);

    for (auto reverseLayerIterator = layers.rbegin(); reverseLayerIterator != layers.rend(); ++reverseLayerIterator) {
        errorGradient = (*reverseLayerIterator)->backwardPropagate(errorGradient);
    }
    return errorGradient;
}

Matrix<double> NeuralNetwork::forward(Matrix<double> &inputData) {
    Matrix<double> layerOutput = inputData;

    for (auto& layer : layers) {
        layerOutput = layer->forwardPropagate(layerOutput);
    }

    return layerOutput;
}

void NeuralNetwork::addLayer(std::shared_ptr<Layer> layer) {
    layers.push_back(layer);
}

void NeuralNetwork::updateNetwork() {
    for (auto& layer : layers) {
        // Dynamic cast to check if the layer is a DenseLayer
        auto denseLayer = std::dynamic_pointer_cast<DenseLayer>(layer);
        if (denseLayer) {
            denseLayer->update(*optimizer, learningRate);
        }
    }
}

double NeuralNetwork::calculateTotalError(Matrix<double> &errorGradient) {
    double totalError = 0.0;
    for (size_t i = 0; i < errorGradient.getRows(); ++i) {
        for (size_t j = 0; j < errorGradient.getCols(); ++j) {
            totalError += errorGradient(i, j);
        }
    }
    return totalError;
}

std::pair<std::vector<Matrix<double>>, std::vector<Matrix<double>>> NeuralNetwork::getParameters() {
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

double NeuralNetwork::getLastErrorValue() {
    return lastErrorValue;
}

Error<double>* NeuralNetwork::getError() const {
    return error;
}

void NeuralNetwork::setLastErrorValue(double errorValue) {
    lastErrorValue = errorValue;
}

std::string NeuralNetwork::returnErrorName() {
    int status = 1;
    const char* name = typeid(*this->error).name();
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };
    return (status==0) ? res.get() : name;
}

std::string NeuralNetwork::returnOptimizerName() {
    int status = 1;
    const char* name = typeid(*this->optimizer).name();
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };
    return (status==0) ? res.get() : name;
}

// Function to save the current parameters of a neural network into a JSON file.
// It is a template function, allowing it to handle different data types for the network parameters.
void NeuralNetwork::save(const std::string& filename) {

}
//
// // Function template to convert a matrix into a JSON object.
// // The template allows this function to work with matrices of different data types.
// nlohmann::json NeuralNetwork::matrixToJson(Matrix<double>& matrix) {
//
// }
//
// // Function to load the neural network parameters from a JSON file.
// // The function is a part of the NeuralNetwork template class, which allows it to handle different data types.
// void NeuralNetwork::load(const std::string& filename) {
//
// }
//
// // Function to convert a JSON array into a matrix.
// // This function is useful for converting stored weights and biases back into their matrix form.
// Matrix<double> NeuralNetwork::jsonToMatrix(const nlohmann::json& json) {
//
// }