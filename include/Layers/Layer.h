#ifndef ML_LAYER_H
#define ML_LAYER_H

#include <Eigen>  // Include Eigen library
#include "../Optimizers/Optimizer.h"

class Layer {
public:
    virtual ~Layer() {}

    // Forward pass through this layer
    virtual Eigen::MatrixXd forwardPropagate(Eigen::MatrixXd& input) = 0;
    // Backward pass through this layer
    virtual Eigen::MatrixXd backwardPropagate(Eigen::MatrixXd& input) = 0;

    // Method to update weights - can be empty for layers without weights
    virtual void update(Optimizer<double>& optimizer, double learningRate) {}

protected:
    Eigen::MatrixXd input;  // Input data for the layer
};

#endif // ML_LAYER_H
