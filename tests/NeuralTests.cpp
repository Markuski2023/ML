#include "../include/Layers/DenseLayer.h"
#include "../include/Layers/ReLU.h"
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
    /*network.addLayer(std::make_shared<DenseLayer<double>>(inputSize, hiddenSize));
    network.addLayer(std::make_shared<ReLU>());
    network.addLayer(std::make_shared<DenseLayer<double>>(hiddenSize, outputSize));


    // Set the optimizer
    SGD<double> optimizer(learningRate);
    network.setOptimizer(&optimizer);

    // Training sample
    Matrix<double> input(1, inputSize);
    input(0, 0) = 0.5;  // Example input

    // Corresponding target value
    Matrix<double> target(1, outputSize);
    target(0, 0) = 1.5;  // Example target

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        Matrix<double> output = network.forward(input);

        // Calculate error (MSE)
        Matrix<double> error = output - target;
        double mse = error(0, 0) * error(0, 0);

        // Print output and error every few epochs
        if (epoch % 1 == 0) {
            std::cout << "Epoch " << epoch << " - Output: " << output(0, 0) << ", MSE: " << mse << std::endl;
        }

        // Backward pass and update weights
        network.backward(output, target);
        network.updateNetworkWeights();
    }

    network.save("parameters.json");
*/
    Matrix<double> input(1, inputSize);
    input(0, 0) = 0.5;  // Example input

    network.load("parameters.json");

    network.save("test.json");

    Matrix<double> output = network.forward(input);
    std::cout << output(0,0) << std::endl;
    return 0;
}

