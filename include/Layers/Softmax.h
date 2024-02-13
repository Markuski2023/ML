#ifndef ML_SOFTMAX_H
#define ML_SOFTMAX_H

#include "ActivationLayer.h"
#include <vector>
#include <cmath>

class Softmax : public ActivationLayer {
private:
    Matrix<double> lastInput;
    Matrix<double> lastOutput;

public:

};

#endif //ML_SOFTMAX_H
