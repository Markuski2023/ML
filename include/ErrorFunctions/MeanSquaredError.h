#ifndef ML_MEANSQUAREDERROR_H
#define ML_MEANSQUAREDERROR_H

#include "Error.h"

template <typename T>
class MeanSquaredError : public Error<T> {
public:
    Matrix<T> calculateError(Matrix<T>& predicted, Matrix<T>& actual) override {
        if (predicted.get_rows() != actual.get_rows() || predicted.get_cols() != actual.get_cols()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        Matrix<T> errorGradient(predicted.get_rows(), predicted.get_cols());
        T totalError = 0;

        // Calculate the sum of squared errors
        for (int i = 0; i < predicted.get_rows(); ++i) {
            for (int j = 0; j < predicted.get_cols(); ++j) {
                T diff = predicted(i, j) - actual(i, j);
                totalError += diff * diff;
                errorGradient(i, j) = 2 * diff; // Gradient of MSE w.r.t the predictions
            }
        }

        // Divide by the total number of elements to get the mean
        totalError /= (predicted.get_rows() * predicted.get_cols());

        return errorGradient; // Return the gradient, not the total error
    }
};


#endif //ML_MEANSQUAREDERROR_H