#ifndef ML_BINARYCROSSENTROPY_H
#define ML_BINARYCROSSENTROPY_H

#include "Error.h"
#include <cmath> // For log function

template <typename T>
class BinaryCrossEntropy : public Error<T> {
public:
    T calculateError(Matrix<T>& predicted, Matrix<T>& actual) {
        if (predicted.get_rows() != actual.get_rows() || predicted.get_cols() != actual.get_cols()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        T sum = 0;
        size_t count = predicted.get_rows() * predicted.get_cols();  // Total elements

        // Calculate the binary cross-entropy
        for (int i = 0; i < predicted.get_rows(); ++i) {
            for (int j = 0; j < predicted.get_cols(); ++j) {
                T p = predicted(i, j);
                T y = actual(i, j);

                // Ensure predicted value p is between 0 and 1
                p = std::max(std::min(p, static_cast<T>(1) - std::numeric_limits<T>::epsilon()), std::numeric_limits<T>::epsilon());

                sum += - (y * std::log(p) + (1 - y) * std::log(1 - p));
            }
        }

        // Calculate the mean
        T bce = sum / count;
        return bce;
    }
};

#endif //ML_BINARYCROSSENTROPY_H