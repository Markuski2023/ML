#ifndef ML_BINARYCROSSENTROPY_H
#define ML_BINARYCROSSENTROPY_H

#include "Error.h"
#include <cmath> // For log function

template <typename T>
class BinaryCrossEntropy : public Error<T> {
public:
    double calculateError(Matrix<T>& predicted, Matrix<T>& actual) override {

    }
};

#endif //ML_BINARYCROSSENTROPY_H