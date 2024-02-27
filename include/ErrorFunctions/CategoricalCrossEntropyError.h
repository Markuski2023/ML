#ifndef ML_CATEGORICALCROSSENTROPYERROR_H
#define ML_CATEGORICALCROSSENTROPYERROR_H

#include "Error.h"
#include <Eigen>

template <typename T>
class CategoricalCrossEntropyError : public Error<T> {
public:
    Matrix<T> calculateError(Eigen::MatrixXd& predicted, Eigen::MatrixXd& actual) override {

    }
};


#endif //ML_CATEGORICALCROSSENTROPYERROR_H
