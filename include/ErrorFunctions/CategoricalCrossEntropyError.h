#ifndef ML_CATEGORICALCROSSENTROPYERROR_H
#define ML_CATEGORICALCROSSENTROPYERROR_H

#include "Error.h"

template <typename T>
class CategoricalCrossEntropyLoss : public Error<T> {
    Matrix<T> calculateError(Matrix<T>& predicted, Matrix<T>& actual) {
        if (predicted.get_rows() != actual.get_rows() || predicted.get_cols() != actual.get_cols()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }


    }
};

#endif //ML_CATEGORICALCROSSENTROPYERROR_H
