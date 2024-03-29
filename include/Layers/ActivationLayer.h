#ifndef ML_ACTIVATIONLAYER_H
#define ML_ACTIVATIONLAYER_H

#include "Layer.h"
#include "../Matrix.h"

// ActivationLayer inherits from Layer and uses Matrix<double> for both forward and backward.
class ActivationLayer : public Layer {
public:
    virtual ~ActivationLayer() override = default;

    // Override these methods to implement specific activation functions
    Matrix<double> forwardPropagate(Matrix<double>& input) override = 0;
    Matrix<double> backwardPropagate(Matrix<double>& output) override = 0;
};

#endif //ML_ACTIVATIONLAYER_H
