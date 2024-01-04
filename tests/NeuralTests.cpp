#include "../include/Layers/DenseLayer.h"
#include "../include/Layers/ReLU.h"
#include "../include/Layers/Sigmoid.h"
#include "../include/ErrorFunctions/MeanSquaredError.h"
#include "../include/ErrorFunctions/BinaryCrossEntropyError.h"
#include "../include/Optimizers/SGD.h"
#include "../include/NeuralNetwork.h"
#include <iostream>

int main() {
    // Network configuration
    const int inputSize = 1;   // Single feature input
    const int hiddenSize = 5;  // Hidden layer size
    const int outputSize = 1;  // Single value output
    const int epochs = 5000;   // Number of training epochs
    const double learningRate = 0.01;

    // Create the neural network
    NeuralNetwork<double> network;
    network.addLayer(std::make_shared<DenseLayer<double>>(inputSize, hiddenSize));
    network.addLayer(std::make_shared<ReLU>());
    network.addLayer(std::make_shared<DenseLayer<double>>(hiddenSize, outputSize));
    network.addLayer(std::make_shared<Sigmoid>());

    // Set the optimizer
    SGD<double> optimizer(learningRate);
    network.setOptimizer(&optimizer);

    BinaryCrossEntropy<double> mse;

    // Training sample
    Matrix<double> input(1, inputSize);
    input(0, 0) = 0.5;  // Example input

    // Corresponding target value
    Matrix<double> target(1, outputSize);
    target(0, 0) = 1;  // Example target

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        Matrix<double> output = network.forward(input);

        // Calculate error (MSE) - corrected to compare output with target
        double error = mse.calculateError(output, target);

        // Print output and error every few epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " - Output: " << output(0, 0) << ", MSE: " << error << std::endl;
        }

        // Backward pass and update weights - ensure backward method is implemented correctly
        network.backward(output, target);
        network.updateNetworkWeights();
    }

    network.save("parameters.json");

    Matrix<double> output = network.forward(input);
    std::cout << output(0,0) << std::endl;
    return 0;
}