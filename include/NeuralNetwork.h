#ifndef ML_NEURALNETWORK_H
#define ML_NEURALNETWORK_H

#include <vector>
#include <memory>
#include "Layers/Layer.h"
#include "Matrix.h"
#include "Optimizers/Optimizer.h"
#include "../json.hpp"

template <typename T>
class NeuralNetwork {
public:
    NeuralNetwork() : optimizer(nullptr), learningRate(0.001) {} // Default learning rate

    void addLayer(std::shared_ptr<Layer> layer);
    Matrix<T> forward(Matrix<T>& input);
    Matrix<T> backward(Matrix<T> &output, Matrix<T> &target);
    void setOptimizer(Optimizer<T>* opt) { optimizer = opt; };
    void setError(Error<T>* err) { error = err; };

    void updateNetworkWeights();
    double calculateTotalError(Matrix<double>& errorGradient);
    T getLastErrorValue();
    void setLastErrorValue(T errorValue);
    std::string returnErrorName();
    std::string returnOptimizerName();

    std::pair<std::vector<Matrix<T>>, std::vector<Matrix<T>>> getWeights();
    void save(const std::string& filename);
    void load(const std::string& filename);
    nlohmann::json matrixToJson(Matrix<T>& matrix);
    Matrix<T> jsonToMatrix(const nlohmann::json& json);

    // To-Do: Methods for predicting, printing summary, etc.

private:
    double learningRate;
    std::vector<std::shared_ptr<Layer>> layers;
    Optimizer<T>* optimizer;
    Error<T>* error;
    T lastErrorValue;
};

#include "../src/NeuralNetwork.tpp"

#endif //ML_NEURALNETWORK_H