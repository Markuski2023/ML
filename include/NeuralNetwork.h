#ifndef ML_NEURALNETWORK_H
#define ML_NEURALNETWORK_H

#include <vector>
#include <memory>
#include "Layers/Layer.h"
#include "Matrix.h"
#include "Optimizers/Optimizer.h"
#include "../json.hpp"
#include "../include/ErrorFunctions/Error.h"

class NeuralNetwork {
public:
    NeuralNetwork() : optimizer(nullptr), learningRate(0.001) {} // Default learning rate

    void addLayer(std::shared_ptr<Layer> layer);
    Matrix<double> forward(Matrix<double>& input);
    Matrix<double> backward(Matrix<double> &output, Matrix<double> &target);
    void setOptimizer(Optimizer<double>* opt) { optimizer = opt; };
    void setError(Error<double>* err) { error = err; };

    void updateNetworkWeights();
    double calculateTotalError(Matrix<double>& errorGradient);
    double getLastErrorValue();
    void setLastErrorValue(double errorValue);
    std::string returnErrorName();
    std::string returnOptimizerName();

    std::pair<std::vector<Matrix<double>>, std::vector<Matrix<double>>> getWeights();
    void save(const std::string& filename);
    void load(const std::string& filename);
    nlohmann::json matrixToJson(Matrix<double>& matrix);
    Matrix<double> jsonToMatrix(const nlohmann::json& json);

    // To-Do: Methods for predicting, printing summary, etc.

private:
    double learningRate;
    std::vector<std::shared_ptr<Layer>> layers;
    Optimizer<double>* optimizer;
    Error<double>* error;
    double lastErrorValue;
};

#endif //ML_NEURALNETWORK_H
