#ifndef ML_RELU_H
#define ML_RELU_H

#include "ActivationLayer.h"
#include <algorithm> // For std::max

// ReLU inherits from ActivationLayer
class ReLU : public ActivationLayer {
public:
    Matrix<double> forwardPropagate(Matrix<double> &input) override {
        for (size_t i = 0; i < input.get_rows(); ++i) {
            for (size_t j = 0; j < input.get_cols(); ++j) {
                input(i, j) = std::max(0.0, input(i, j));
            }
        }
        return input;
    }

    Matrix<double> backwardPropagate(Matrix<double> &input) override {
        for (size_t i = 0; i < input.get_rows(); ++i) {
            for (size_t j = 0; j < input.get_cols(); ++j) {
                if (input(i, j) < 0) {input(i, j) = 0;}
                else {input(i, j) = 1;}
            }
        }
        return input;
    }
};

#endif //ML_RELU_H
