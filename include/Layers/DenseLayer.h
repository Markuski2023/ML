#ifndef ML_DENSELAYER_H
#define ML_DENSELAYER_H

#include "../Matrix.h"
#include "Layer.h"
#include "../Optimizers/Optimizer.h"

class DenseLayer : public Layer {
public:
    DenseLayer();
    DenseLayer(unsigned inputSize, unsigned outputSize);
    Matrix<double> forwardPropagate(Matrix<double>& input) override;
    Matrix<double> backwardPropagate(Matrix<double>& input) override;
    void update(Optimizer<double>& optimizer, double learningRate);

    Matrix<double>& getWeights();
    Matrix<double>& getBiases();

private:
    Matrix<double> weights, biases;
    Matrix<double> storedWeightGradients, storedBiasGradients;
    Matrix<double> input;
};

#endif // ML_DENSELAYER_H