#ifndef ML_OPTIMIZER_H
#define ML_OPTIMIZER_H

#include "../Matrix.h"

template <typename T>
class Optimizer {
public:
    virtual void update(Matrix<T>& weights, Matrix<T>& biases,
                        Matrix<T>& weightGradients, Matrix<T>& biasGradients,
                        double learningRate) = 0;
};

#endif //ML_OPTIMIZER_H
