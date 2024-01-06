#ifndef ML_SOFTMAX_H
#define ML_SOFTMAX_H

#include "ActivationLayer.h"
#include <vector>
#include <cmath>

class Softmax : public ActivationLayer {
private:
    Matrix<double> lastInput;
    Matrix<double> lastOutput;

public:
    // Forward pass
    Matrix<double> forward(Matrix<double>& inputData) override {
        lastInput = inputData; // Storing the input for use in backward pass
        Matrix<double> output(inputData.get_rows(), inputData.get_cols());

        for (size_t i = 0; i < inputData.get_rows(); ++i) {
            double sum = 0;
            for (size_t j = 0; j < inputData.get_cols(); ++j) {
                sum += exp(inputData(i, j));
            }
            for (size_t j = 0; j < inputData.get_cols(); ++j) {
                output(i, j) = exp(inputData(i, j)) / sum;
            }
        }
        lastOutput = output; // Saving the output for use in backward pass
        return output;
    }

    // Backward pass
    Matrix<double> backward(Matrix<double>& outputError) override {
        Matrix<double> inputError(lastInput.get_rows(), lastInput.get_cols());

        for (size_t i = 0; i < lastInput.get_rows(); ++i) {
            for (size_t j = 0; j < lastInput.get_cols(); ++j) {
                double derivative = lastOutput(i, j) * (1 - lastOutput(i, j));
                inputError(i, j) = derivative * outputError(i, j);
            }
        }
        return inputError;
    }
};

#endif //ML_SOFTMAX_H
