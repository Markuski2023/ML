#ifndef ML_DENSELAYER_H
#define ML_DENSELAYER_H

#include <Eigen>  // Include Eigen library
#include "Layer.h"
#include "../Optimizers/Optimizer.h"

class DenseLayer : public Layer {
public:
    DenseLayer();
    DenseLayer(unsigned inputSize, unsigned outputSize);
    Eigen::MatrixXd forwardPropagate(Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backwardPropagate(Eigen::MatrixXd& input) override;
    void update(Optimizer<double>& optimizer, double learningRate);

    Eigen::MatrixXd& getWeights();
    Eigen::MatrixXd& getBiases();

private:
    Eigen::MatrixXd weights, biases;
    Eigen::MatrixXd storedWeightGradients, storedBiasGradients;
    Eigen::MatrixXd input;  // Input data for the layer
};

#endif // ML_DENSELAYER_H
