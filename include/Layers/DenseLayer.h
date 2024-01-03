#ifndef ML_DENSELAYER_H
#define ML_DENSELAYER_H

#include "../Matrix.h"
#include "Layer.h"
#include "../Optimizers/Optimizer.h"
#include <memory>
#include <vector>
#include <random>

template <typename T>
class DenseLayer : public Layer {
public:
    DenseLayer();
    DenseLayer(unsigned inputSize, unsigned outputSize);
    Matrix<T> forward(Matrix<T>& input) override;
    Matrix<T> backward(Matrix<T>& outputError) override;
    void updateWeights(Optimizer<T>& optimizer, double learningRate);

    Matrix<T>& getWeights();
    Matrix<T>& getBiases();

    void setWeights(Matrix<T>& newWeights);
    void setBiases(Matrix<T>& newBiases);

private:
    Matrix<T> weights, biases;
    Matrix<T> weightsError, biasesError;
    Matrix<T> input;
};

#include "../../src/DenseLayer.tpp"

#endif // ML_DENSELAYER_H