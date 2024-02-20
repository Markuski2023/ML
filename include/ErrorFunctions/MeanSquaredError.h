#ifndef ML_MEANSQUAREDERROR_H
#define ML_MEANSQUAREDERROR_H

#include "Error.h"

template <typename T>
class MeanSquaredError : public Error<T> {
public:
    Matrix<T> calculateError(Matrix<T>& predicted, Matrix<T>& actual) override {
        Matrix<T> errorGradient(predicted.getRows(), predicted.getCols());

        // Calculate the gradient of MSE with respect to the predicted output
        for (size_t i = 0; i < predicted.getRows(); ++i) {
            for (size_t j = 0; j < predicted.getCols(); ++j) {
                // MSE derivative: 2 * (predicted - actual) / N
                errorGradient(i, j) = 2 * (predicted(i, j) - actual(i, j)) / predicted.getRows();
            }
        }
        return errorGradient;
    }
};


#endif //ML_MEANSQUAREDERROR_H