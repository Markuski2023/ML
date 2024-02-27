#ifndef ML_BINARYCROSSENTROPY_H
#define ML_BINARYCROSSENTROPY_H

#include "Error.h"
#include <cmath> // For log function

template <typename T>
class BinaryCrossEntropy : public Error<T> {
public:
    double calculateError(Eigen::MatrixXd& predicted, Eigen::MatrixXd& actual) override {

    }
};

#endif //ML_BINARYCROSSENTROPY_H