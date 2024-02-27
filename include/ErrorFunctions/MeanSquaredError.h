#ifndef ML_MEANSQUAREDERROR_H
#define ML_MEANSQUAREDERROR_H

#include "Error.h"
#include <Eigen>

template <typename T>
class MeanSquaredError : public Error<T> {
public:
    Eigen::MatrixXd calculateError(Eigen::MatrixXd& predicted, Eigen::MatrixXd& actual) override {
        // Ensure that predicted and actual matrices are of the same size
        if (predicted.rows() != actual.rows() || predicted.cols() != actual.cols()) {
            throw std::invalid_argument("Predicted and actual matrices must be of the same size.");
        }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> errorGradient =
            2 * (predicted - actual) / predicted.rows();

        return errorGradient;
    }
};

#endif // ML_MEANSQUAREDERROR_H
