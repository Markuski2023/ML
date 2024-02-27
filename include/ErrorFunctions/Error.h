#ifndef ML_ERROR_H
#define ML_ERROR_H
#include "../Matrix.h"
#include <Eigen>

template <typename T>
class Error {
public:
    virtual ~Error() = default;

    virtual Matrix<T> calculateError(Eigen::MatrixXd &predicted, Eigen::MatrixXd &actual) = 0;
};

#endif //ML_ERROR_H