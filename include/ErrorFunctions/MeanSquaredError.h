#ifndef ML_MEANSQUAREDERROR_H
#define ML_MEANSQUAREDERROR_H

#include "Error.h"

template <typename T>
class MeanSquaredError : public Error<T> {
public:
    Matrix<T> calculateError(Matrix<T>& predicted, Matrix<T>& actual) override {

    }
};


#endif //ML_MEANSQUAREDERROR_H