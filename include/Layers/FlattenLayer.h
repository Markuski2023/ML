#ifndef FLATTENLAYER_H
#define FLATTENLAYER_H

#include "Layer.h"
#include "../Matrix.h"

template <typename T>
class FlattenLayer : public Layer<T> {

    Matrix<double> forward(std::vector<Matrix<T>>& input) {
            if (input.empty()) {
                throw std::runtime_error("Input vector is empty");
            }

            // Assuming all matrices in the vector have the same dimensions
            const size_t rowsPerMatrix = input[0].get_rows();
            const size_t colsPerMatrix = input[0].get_cols();
            const size_t totalElementsPerMatrix = rowsPerMatrix * colsPerMatrix;

            // Create a matrix to store the flattened matrices
            Matrix<double> flatten(input.size(), totalElementsPerMatrix);

            for (size_t matrixIndex = 0; matrixIndex < input.size(); ++matrixIndex) {
                Matrix<T>& currentMatrix = input[matrixIndex];
                size_t count = 0; // Reset count for each matrix

                for (size_t m = 0; m < currentMatrix.get_rows(); ++m) {
                    for (size_t n = 0; n < currentMatrix.get_cols(); ++n) {
                        flatten(matrixIndex, count) = currentMatrix(m, n);
                        ++count;
                    }
                }
            }
            return flatten;
        }

    std::vector<Matrix<double>> backward(Matrix<double>& outputGradient) {
        std::vector<Matrix<double>> inputGradient;

        for (size_t i = 0; i < outputGradient.get_rows(); ++i) {
            // Initialize a matrix with the shape of the original input
            Matrix<double> reshapedGradient(inputRows, inputCols);

            for (size_t r = 0; r < inputRows; ++r) {
                for (size_t c = 0; c < inputCols; ++c) {
                    // Map the 1D gradient back to the 2D form
                    reshapedGradient(r, c) = outputGradient(i, r * inputCols + c);
                }
            }

            inputGradient.push_back(reshapedGradient);
        }

        return inputGradient;
    }
};

#endif //FLATTENLAYER_H