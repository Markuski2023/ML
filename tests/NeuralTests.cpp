#include "../include/Layers/DenseLayer.h"
#include "../include/Layers/ReLU.h"
#include "../include/ErrorFunctions/MeanSquaredError.h"
#include "../include/Optimizers/SGD.h"
#include "../include/NeuralNetwork.h"
#include <iostream>

// Assume a function to calculate total error is defined
double calculateTotalError(Matrix<double>& errorGradient);

int main() {
    // Network configuration
    constexpr int inputSize = 3;   // Adjust for your input features (now 3)
    constexpr int hiddenSize = 5;  // Hidden layer size
    constexpr int outputSize = 4;  // Output neurons (now 4)
    constexpr int epochs = 5000;   // Number of training epochs
    constexpr double learningRate = 0.01;

    // Create the neural network
    NeuralNetwork<double> network;
    network.addLayer(std::make_shared<DenseLayer<double>>(inputSize, hiddenSize));
    network.addLayer(std::make_shared<ReLU>());
    network.addLayer(std::make_shared<DenseLayer<double>>(hiddenSize, outputSize));

    // Set the optimizer
    SGD<double> optimizer(learningRate);
    network.setOptimizer(&optimizer);

    // Set the error function
    MeanSquaredError<double> mse;
    network.setError(&mse);

    // Example training data for regression
    Matrix<double> input(1, inputSize); // Input size is now 3
    // Set your input values here for 3 inputs
    input(0, 0) = 0.5;
    input(0, 1) = -0.5;
    input(0, 2) = 0.25;

    // Corresponding target values for regression
    Matrix<double> target(1, outputSize); // Output size is now 4
    // Set your target values here (for 4 output neurons)
    target(0, 0) = 1.5;  // Example target value for first neuron
    target(0, 1) = 2.0;  // Example target value for second neuron
    target(0, 2) = 0.5;  // Example target value for third neuron
    target(0, 3) = 1.0;  // Example target value for fourth neuron

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        Matrix<double> output = network.forward(input);

        // Calculate error
        Matrix<double> errorGradient = mse.calculateError(output, target);
        network.setLastErrorValue(calculateTotalError(errorGradient));

        // Print output and error every few epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " - Output: ";
            for (int i = 0; i < outputSize; ++i) {
                std::cout << output(0, i) << " ";
            }
            std::cout << ", MSE: " << network.getLastErrorValue() << std::endl;
        }

        // Backward pass and update weights
        network.backward(output, target);
        network.updateNetworkWeights();
    }

    // Save the trained network parameters
    network.save("parameters.json");

    // Test the network with the input data
    Matrix<double> finalOutput = network.forward(input);
    for (int i = 0; i < outputSize; ++i) {
        std::cout << finalOutput(0, i) << " ";
    }
    std::cout << std::endl;

    // Display the optimizer and error function used
    std::cout << network.returnOptimizerName() << std::endl;
    std::cout << network.returnErrorName() << std::endl;

    return 0;
}
