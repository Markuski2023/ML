#include "../include/NeuralNetwork.h"
#include "../include/Layers/DenseLayer.h"
#include <filesystem>
#include <string>
#include <typeinfo>
#include <cxxabi.h>
#include <memory>
#include "../json.hpp"
#include <Eigen>  // Changed to include Eigen

Eigen::MatrixXd NeuralNetwork::backward(Eigen::MatrixXd &networkOutput, Eigen::MatrixXd &expectedOutput) {
    Eigen::MatrixXd errorGradient = error->calculateError(networkOutput, expectedOutput);

    for (auto reverseLayerIterator = layers.rbegin(); reverseLayerIterator != layers.rend(); ++reverseLayerIterator) {
        errorGradient = (*reverseLayerIterator)->backwardPropagate(errorGradient);
    }
    return errorGradient;
}

Eigen::MatrixXd NeuralNetwork::forward(Eigen::MatrixXd &inputData) {
    Eigen::MatrixXd layerOutput = inputData;

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
        auto denseLayer = std::dynamic_pointer_cast<DenseLayer>(layer);
        if (denseLayer) {
            denseLayer->update(*optimizer, learningRate);
        }
    }
}

double NeuralNetwork::calculateTotalError(Eigen::MatrixXd &errorGradient) {
    double totalError = 0.0;
    for (int i = 0; i < errorGradient.rows(); ++i) {
        for (int j = 0; j < errorGradient.cols(); ++j) {
            totalError += errorGradient(i, j);
        }
    }
    return totalError;
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> NeuralNetwork::getParameters() {
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::MatrixXd> biases;

    for (auto& layer : layers) {
        auto denseLayer = std::dynamic_pointer_cast<DenseLayer>(layer);
        if (denseLayer) {
            weights.push_back(denseLayer->getWeights());
            biases.push_back(denseLayer->getBiases());
        }
    }
    return std::pair(weights, biases);
}

// Other functions remain the same
