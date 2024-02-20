#ifndef ML_ERROR_H
#define ML_ERROR_H
#include "../Matrix.h"

template <typename T>
class Error {
public:
    virtual ~Error() = default;

    virtual Matrix<T> calculateError(Matrix<T>& predicted, Matrix<T>& actual) = 0;
};

#endif //ML_ERROR_H