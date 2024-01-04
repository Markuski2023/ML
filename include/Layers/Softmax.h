#ifndef ML_SOFTMAX_H
#define ML_SOFTMAX_H

#include "ActivationLayer.h"

class Softmax : public ActivationLayer {
    Matrix<double> forward(Matrix<double>& input) override {

    }

    Matrix<double> backward(Matrix<double>& outputError) override {

    }
};

#endif //ML_SOFTMAX_H