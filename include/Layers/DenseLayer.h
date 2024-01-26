#ifndef ML_DENSELAYER_H
#define ML_DENSELAYER_H

#include "../Matrix.h"
#include "Layer.h"
#include "../Optimizers/Optimizer.h"

class DenseLayer : public Layer {
public:
    DenseLayer();
    DenseLayer(unsigned inputSize, unsigned outputSize);
    Matrix<double> forward(Matrix<double>& input) override;
    Matrix<double> backward(Matrix<double>& outputError) override;
    void updateWeights(Optimizer<double>& optimizer, double learningRate);

    Matrix<double>& getWeights();
    Matrix<double>& getBiases();

    void setWeights(Matrix<double>& newWeights);
    void setBiases(Matrix<double>& newBiases);

private:
    Matrix<double> weights, biases;
    Matrix<double> weightsError, biasesError;
    Matrix<double> input;
};

#endif // ML_DENSELAYER_H