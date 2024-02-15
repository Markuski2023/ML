#ifndef ML_CATEGORICALCROSSENTROPYERROR_H
#define ML_CATEGORICALCROSSENTROPYERROR_H

#include "Error.h"
#include <cmath>  // Include for log function

template <typename T>
class CategoricalCrossEntropyError : public Error<T> {
public:
    Matrix<T> calculateError(Matrix<T>& predicted, Matrix<T>& actual) override {

    }
};


#endif //ML_CATEGORICALCROSSENTROPYERROR_H
