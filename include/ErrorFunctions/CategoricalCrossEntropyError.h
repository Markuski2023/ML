#ifndef ML_CATEGORICALCROSSENTROPYERROR_H
#define ML_CATEGORICALCROSSENTROPYERROR_H

#include "Error.h"
#include <cmath>  // Include for log function

template <typename T>
class CategoricalCrossEntropyLoss : public Error<T> {
public:
    T calculateError(Matrix<T>& predicted, Matrix<T>& actual) override {
        if (predicted.get_rows() != actual.get_rows() || predicted.get_cols() != actual.get_cols()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        T total_loss = 0;
        size_t count = predicted.get_rows();  // Number of samples

        for (size_t i = 0; i < count; ++i) {
            T sample_loss = 0;
            for (size_t j = 0; j < predicted.get_cols(); ++j) {
                sample_loss += -actual(i, j) * std::log(predicted(i, j) + static_cast<T>(10e-7));
            }

            total_loss += sample_loss;
        }
        return total_loss / count;
    }
};

#endif //ML_CATEGORICALCROSSENTROPYERROR_H
