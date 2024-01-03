#ifndef ML_MEANSQUAREDERROR_H
#define ML_MEANSQUAREDERROR_H

#include "Error.h"

template <typename T>
class MeanSquaredError : public Error<T> {
public:
    T calculateError(Matrix<T>& predicted, Matrix<T>& actual) {
        Matrix<T> error = predicted - actual;
        T sum = 0;
        int count = 0;

        // Calculate the sum of squared errors
        for (int i = 0; i < error.rows(); ++i) {
            for (int j = 0; j < error.cols(); ++j) {
                sum += error(i, j) * error(i, j);
                ++count;
            }
        }

        // Calculate the mean
        T mse = sum / static_cast<T>(count);
        return mse;
    }
};


#endif //ML_MEANSQUAREDERROR_H
