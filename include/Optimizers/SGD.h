#ifndef ML_SGD_H
#define ML_SGD_H

#include "Optimizer.h"

template <typename T>
class SGD : public Optimizer<T> {
public:
    explicit SGD(double lr = 0.01) : learningRate(lr) {}

    void update(Matrix<T>& weights, Matrix<T>& biases, Matrix<T>& weightGradients, Matrix<T>& biasGradients, double learningRate) override {
        for (size_t i = 0; i < weights.get_rows(); ++i) {
            for (size_t j = 0; j < weights.get_cols(); ++j) {
                weights(i, j) -= learningRate * weightGradients(i, j);
            }
        }

        for (size_t i = 0; i < biases.get_rows(); ++i) {
            for (size_t j = 0; j < biases.get_cols(); ++j) {
                biases(i, j) -= learningRate * biasGradients(i, j);
            }
        }
    }
private:
    double learningRate;
};


#endif //ML_SGD_H
