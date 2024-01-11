#include "../include/Layers/DenseLayer.h"
#include "../include/Layers/ReLU.h"
#include "../include/ErrorFunctions/MeanSquaredError.h"  // Assuming you have MSE implemented
#include "../include/Optimizers/SGD.h"
#include "../include/NeuralNetwork.h"
#include <iostream>

int main() {
    // Network configuration
    const int inputSize = 2;    // Adjust for your input features
    const int hiddenSize = 5;   // Hidden layer size
    const int outputSize = 1;   // Single output for regression
    const int epochs = 5000;    // Number of training epochs
    const double learningRate = 0.01;

    // Create the neural network
    NeuralNetwork<double> network;
    network.addLayer(std::make_shared<DenseLayer<double>>(inputSize, hiddenSize));
    network.addLayer(std::make_shared<ReLU>());
    network.addLayer(std::make_shared<DenseLayer<double>>(hiddenSize, outputSize));

    // Set the optimizer
    SGD<double> optimizer(learningRate);
    network.setOptimizer(&optimizer);

    // Set the error
    MeanSquaredError<double> mse;
    network.setError(&mse);

    // Example training data for regression
    Matrix<double> input(1, inputSize);
    // Set your input values here
    input(0, 0) = 0.5;
    input(0, 1) = 0.3;

    // Corresponding target value for regression
    Matrix<double> target(1, outputSize);
    target(0, 0) = 1.2;  // Example target value

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        Matrix<double> output = network.forward(input);

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
    network.save("parameters.json");

    Matrix<double> finalOutput = network.forward(input);
    for (int i = 0; i < outputSize; ++i) {
        std::cout << finalOutput(0, i) << " ";
    }
    std::cout << std::endl;

    std::cout << network.returnOptimizerName() << std::endl;
    std::cout << network.returnErrorName() << std::endl;

    return 0;
}
