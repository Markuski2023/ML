#ifndef ML_NEURALNETWORK_H
#define ML_NEURALNETWORK_H

#include <vector>
#include <memory>
#include <Eigen>  // Include Eigen library
#include "Layers/Layer.h"
#include "Optimizers/Optimizer.h"
#include "../json.hpp"
#include "../include/ErrorFunctions/Error.h"

class NeuralNetwork {
public:
    NeuralNetwork() : optimizer(nullptr), learningRate(0.001) {} // Default learning rate

    void addLayer(std::shared_ptr<Layer> layer);
    Eigen::MatrixXd forward(Eigen::MatrixXd& input);
    Eigen::MatrixXd backward(Eigen::MatrixXd &output, Eigen::MatrixXd &target);
    void setOptimizer(Optimizer<double>* opt) { optimizer = opt; };
    void setError(Error<double>* err) { error = err; };

    void updateNetwork();
    double calculateTotalError(Eigen::MatrixXd& errorGradient);
    double getLastErrorValue();
    void setLastErrorValue(double errorValue);
    std::string returnErrorName();
    std::string returnOptimizerName();
    Error<double>* getError() const;

    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> getParameters();
    void save(const std::string& filename);
    void load(const std::string& filename);
    nlohmann::json matrixToJson(Eigen::MatrixXd& matrix);
    Eigen::MatrixXd jsonToMatrix(const nlohmann::json& json);

    // To-Do: Methods for predicting, printing summary, etc.

private:
    double learningRate;
    std::vector<std::shared_ptr<Layer>> layers;
    Optimizer<double>* optimizer;
    Error<double>* error;
    double lastErrorValue;
};

#endif // ML_NEURALNETWORK_H
