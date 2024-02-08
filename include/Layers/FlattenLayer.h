#ifndef FLATTENLAYER_H
#define FLATTENLAYER_H

#include "Layer.h"
#include "../Matrix.h"

class FlattenLayer : public Layer {
public:
    Matrix<double> forward(std::vector<Matrix<double>> &input) {
        Matrix<double> output(input.size(), (input[0].get_rows() * input[0].get_cols()));
        for (size_t i = 0; i < input.size(); ++i) {
            for (size_t j = 0; j < input[0].get_cols(); ++j) {
                for (size_t k = 0; k < input[0].get_rows(); ++k) {
                    output[i][j * input[0].get_rows() + k] = input[i][j][k];
                }
            }
        }
        return output;
    }

    Matrix<double> backwardPropagate(Matrix<double> &input) {

    }
private:

};

#endif //FLATTENLAYER_H