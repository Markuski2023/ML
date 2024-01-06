#include "../include/Layers/DenseLayer.h"
#include "../include/Layers/ReLU.h"
#include "../include/Layers/Sigmoid.h"
#include "../include/Layers/Softmax.h"
#include "../include/ErrorFunctions/BinaryCrossEntropyError.h"
#include "../include/ErrorFunctions/CategoricalCrossEntropyError.h" // Include Cross-Entropy Error function
#include "../include/Optimizers/SGD.h"
#include "../include/NeuralNetwork.h"
#include <iostream>

int main() {
    // Network configuration
    const int inputSize = 2;   // Adjust for a more complex input, e.g., 2 features
    const int hiddenSize = 5;  // Hidden layer size
    const int outputSize = 3;  // Adjust for multiple output classes, e.g., 3 classes
    const int epochs = 5000;   // Number of training epochs
    const double learningRate = 0.01;

    // Create the neural network
    NeuralNetwork<double> network;
    network.addLayer(std::make_shared<DenseLayer<double>>(inputSize, hiddenSize));
    network.addLayer(std::make_shared<ReLU>());
    network.addLayer(std::make_shared<DenseLayer<double>>(hiddenSize, outputSize));
    network.addLayer(std::make_shared<Softmax>());

    // Set the optimizer
    SGD<double> optimizer(learningRate);
    network.setOptimizer(&optimizer);

    CategoricalCrossEntropyLoss<double> crossEntropy; // Using Cross-Entropy Error for softmax

    // Example training data for a classification task
    Matrix<double> input(1, inputSize);
    // Set your input values here
    input(0, 0) = 0.5;
    input(0, 1) = 0.3;

    // Corresponding target value for multi-class (one-hot encoded)
    Matrix<double> target(1, outputSize);
    target(0, 0) = 0;  // Class 1
    target(0, 1) = 1;  // Class 2
    target(0, 2) = 0;  // Class 3

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        Matrix<double> output = network.forward(input);

        // Calculate error (Cross-Entropy)
        double error = crossEntropy.calculateError(output, target);

        // Print output and error every few epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " - Output: ";
            for (int i = 0; i < outputSize; ++i) {
                std::cout << output(0, i) << " ";
            }
            std::cout << ", Cross-Entropy: " << error << std::endl;
        }

        // Backward pass and update weights
        network.backward(output, target);
        network.updateNetworkWeights();
    }

    network.save("parameters.json");

    Matrix<double> finalOutput = network.forward(input);
    for (int i = 0; i < outputSize; ++i) {
        std::cout << finalOutput(0, i) << " ";
    }
    std::cout << std::endl;

    return 0;
}