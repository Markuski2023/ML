#ifndef ML_SIGMOID_H
#define ML_SIGMOID_H

#include "ActivationLayer.h"
#include <cmath>

class Sigmoid : public ActivationLayer {
private:
    Matrix<double> lastOutput;  // Store the last output of the forward pass

public:

};


#endif //ML_SIGMOID_H