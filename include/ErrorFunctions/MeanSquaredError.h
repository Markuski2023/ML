#ifndef ML_MEANSQUAREDERROR_H
#define ML_MEANSQUAREDERROR_H

#include "Error.h"

template <typename T>
class MeanSquaredError : public Error<T> {
public:
    T calculateError(Matrix<T>& predicted, Matrix<T>& actual) {
        if (predicted.get_rows() != actual.get_rows() || predicted.get_cols() != actual.get_cols()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        T sum = 0;
        size_t count = predicted.get_rows() * predicted.get_cols();

        // Calculate the sum of squared errors
        for (int i = 0; i < predicted.get_rows(); ++i) {
            for (int j = 0; j < predicted.get_cols(); ++j) {
                T diff = predicted(i, j) - actual(i, j);
                sum += diff * diff;
            }
        }

        // Calculate the mean
        T mse = sum / count;
        return mse;
    }
};

#endif //ML_MEANSQUAREDERROR_H