#include "../include/Layers/DenseLayer.h"
#include "../include/Layers/ReLU.h"
#include "../include/ErrorFunctions/MeanSquaredError.h"
#include "../include/Optimizers/SGD.h"
#include "../include/NeuralNetwork.h"
#include <iostream>

// Function to generate random double between min and max
double randomDouble(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Create input and target data
void createData(Matrix<double>& input, Matrix<double>& target) {
    // Define weights for the synthetic relation
    std::vector<double> weights = {0.2, 0.4, 0.1, 0.5, 0.3, 0.7, 0.6, 0.2, 0.4, 0.1};

    for (size_t i = 0; i < input.getRows(); ++i) {
        double targetValue = 0.0;

        for (size_t j = 0; j < input.getCols(); ++j) {
            // Generate random input
            input(i, j) = randomDouble(0.0, 1.0);

            // Calculate target value as a weighted sum plus some noise
            targetValue += weights[j] * input(i, j);
        }
        // Add random noise to target
        target(i, 0) = targetValue + randomDouble(-0.1, 0.1);
    }
}

void createBatches(std::vector<Matrix<double>>& inputs,
                   std::vector<Matrix<double>>& targets,
                   int batchSize, int numBatches) {
    for (int batch = 0; batch < numBatches; ++batch) {
        Matrix<double> input(batchSize, 10);   // Batch size, input size
        Matrix<double> target(batchSize, 1);   // Batch size, output size

        createData(input, target);  // Use your existing createData function

        inputs.push_back(input);
        targets.push_back(target);
    }
}


int main() {
    // Network configuration
    constexpr int inputSize = 10;   // Input features
    constexpr int hiddenSize = 5;   // Hidden layer size
    constexpr int outputSize = 1;   // Output neurons
    constexpr int epochs = 100;    // Number of training epochs
    constexpr double learningRate = 0.01;

    // Create the neural network
    NeuralNetwork network;
    network.addLayer(std::make_shared<DenseLayer>(inputSize, hiddenSize));
    network.addLayer(std::make_shared<ReLU>());
    network.addLayer(std::make_shared<DenseLayer>(hiddenSize, outputSize));

    // Set the optimizer
    SGD<double> optimizer(learningRate);
    network.setOptimizer(&optimizer);

    // Set the error function
    MeanSquaredError<double> mse;
    network.setError(&mse);

    constexpr int batchSize = 2;   // Size of each batch
    constexpr int numBatches = 5;  // Number of batches

    std::vector<Matrix<double>> inputs;
    std::vector<Matrix<double>> targets;

    // Create multiple batches of data
    createBatches(inputs, targets, batchSize, numBatches);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;

        for (int batch = 0; batch < numBatches; ++batch) {
            // Forward pass
            Matrix<double> output = network.forward(inputs[batch]);

            // Calculate error
            Matrix<double> errorGradient = mse.calculateError(output, targets[batch]);

            totalError += network.calculateTotalError(errorGradient);


            // Backward pass and update weights
            network.backward(output, targets[batch]);
            network.updateNetwork();

        }

        totalError /= numBatches; // Average error over batches
        network.setLastErrorValue(totalError);

        // Print average error every few epochs
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " - Avg MSE: " << totalError << std::endl;
        }
    }

    std::cout << network.returnErrorName() << std::endl;
    std::cout << network.returnOptimizerName();

    return 0;
}