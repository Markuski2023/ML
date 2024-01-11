#ifndef ML_LAYER_H
#define ML_LAYER_H

#include "../Matrix.h"

class Layer {
public:
    virtual ~Layer() {}

    // Forward pass through this layer
    virtual Matrix<double> forward(Matrix<double>& input) = 0;
    // Backward pass through this layer
    virtual Matrix<double> backward(Matrix<double>& outputError) = 0;

    // Method to update weights - can be empty for layers without weights
    virtual void updateWeights(double learningRate) {}

protected:
    Matrix<double> input;
};


#endif //ML_LAYER_H
