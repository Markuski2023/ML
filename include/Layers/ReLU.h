#ifndef ML_RELU_H
#define ML_RELU_H

#include "ActivationLayer.h"
#include <algorithm> // For std::max

// ReLU inherits from ActivationLayer
class ReLU : public ActivationLayer {
public:
    // Implement the forward function
    Matrix<double> forward(Matrix<double>& input) override {
        Matrix<double> output(input.get_rows(), input.get_cols());
        for (size_t i = 0; i < input.get_rows(); ++i) {
            for (size_t j = 0; j < input.get_cols(); ++j) {
                output(i, j) = std::max(0.0, input(i, j));
            }
        }
        return output;
    }

    // Implement the backward function
    Matrix<double> backward(Matrix<double>& outputError) override {
        Matrix<double> inputError(outputError.get_rows(), outputError.get_cols());
        for (size_t i = 0; i < inputError.get_rows(); ++i) {
            for (size_t j = 0; j < inputError.get_cols(); ++j) {
                inputError(i, j) = outputError(i, j) > 0.0 ? 1.0 : 0.0;
            }
        }
        return inputError;
    }
};

#endif //ML_RELU_H
