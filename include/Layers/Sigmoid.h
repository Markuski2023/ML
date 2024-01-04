#ifndef ML_SIGMOID_H
#define ML_SIGMOID_H

#include "ActivationLayer.h"
#include <cmath>

class Sigmoid : public ActivationLayer {
private:
    Matrix<double> lastOutput;  // Store the last output of the forward pass

public:
    Matrix<double> forward(Matrix<double>& input) override {
        lastOutput = Matrix<double>(input.get_rows(), input.get_cols());

        for (int i = 0; i < input.get_rows(); ++i) {
            for (int j = 0; j < input.get_cols(); ++j) {
                lastOutput(i, j) = 1 / (1 + exp(-input(i, j)));
            }
        }
        return lastOutput;
    }

    Matrix<double> backward(Matrix<double>& outputError) override {
        Matrix<double> inputError = Matrix<double>(outputError.get_rows(), outputError.get_cols());
        for (size_t i = 0; i < inputError.get_rows(); ++i) {
            for (size_t j = 0; j < inputError.get_cols(); ++j) {
                double sigmoidOutput = lastOutput(i, j);
                inputError(i, j) = sigmoidOutput * (1 - sigmoidOutput) * outputError(i, j);
            }
        }
        return inputError;
    }
};


#endif //ML_SIGMOID_H