#ifndef ML_CATEGORICALCROSSENTROPYERROR_H
#define ML_CATEGORICALCROSSENTROPYERROR_H

#include "Error.h"
#include <cmath>  // Include for log function

template <typename T>
class CategoricalCrossEntropyError : public Error<T> {
public:
    Matrix<T> calculateError(Matrix<T>& predicted, Matrix<T>& actual) override {
        if (predicted.get_rows() != actual.get_rows() || predicted.get_cols() != actual.get_cols()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        Matrix<T> errorGradient(predicted.get_rows(), predicted.get_cols());

        for (size_t i = 0; i < predicted.get_rows(); ++i) {
            for (size_t j = 0; j < predicted.get_cols(); ++j) {
                // Assuming the derivative of cross-entropy error for each element
                errorGradient(i, j) = -actual(i, j) / (predicted(i, j) + 1e-7);
            }
        }
        return errorGradient;
    }
};


#endif //ML_CATEGORICALCROSSENTROPYERROR_H
