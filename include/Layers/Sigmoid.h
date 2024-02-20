#ifndef ML_SIGMOID_H
#define ML_SIGMOID_H

#include "ActivationLayer.h"
#include <cmath>

class Sigmoid : public ActivationLayer {
public:
    Matrix<double> forwardPropagate(Matrix<double> &input) override {
        for (size_t i = 0; i < input.getRows(); ++i) {
            for (size_t j = 0; j < input.getCols(); ++j) {
                input(i, j) = 1/(1 + exp(-input(i, j)));
            }
        }
        return input;
    }

    Matrix<double> backwardPropagate(Matrix<double> &input) override {
        for (size_t i = 0; i < input.getRows(); ++i) {
            for (size_t j = 0; j < input.getCols(); ++j) {
                input(i, j) = 1 - (1/(1 + exp(-input(i, j))));
            }
        }
        return input;
    }
};

#endif //ML_SIGMOID_H