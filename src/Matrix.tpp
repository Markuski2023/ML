#include "../include/Matrix.h"

#include <omp.h>
#include <stdexcept>
#include <random>

// Default constructor
template <typename T>
Matrix<T>::Matrix() : rows(0), cols(0) {}

// Parameterized constructor
template <typename T>
Matrix<T>::Matrix(unsigned rows, unsigned cols, const T& initial) : rows(rows), cols(cols), mat(rows, std::vector<T>(cols, initial)) {}

template <typename T>
Matrix<T>::Matrix(unsigned rows, unsigned cols) : rows(rows), cols(cols), mat(rows, std::vector<T>(cols)) {
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> distr(-0.01, 0.01); // Define the range

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat[i][j] = static_cast<T>(distr(gen)); // Assign a random value to variable
        }
    }
}

// Copy constructor
template <typename T>
Matrix<T>::Matrix(const Matrix<T>& rhs) : rows(rhs.rows), cols(rhs.cols), mat(rhs.mat) {}

// Destructor
template <typename T>
Matrix<T>::~Matrix() {}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& rhs) {
    // Self-assignment check: if rhs matrix is equal to the current, return the current
    if(this == &rhs) {
        return *this;
    }

    rows = rhs.rows;
    cols = rhs.cols;
    mat = rhs.mat;

    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& rhs) {
    if(this->rows != rhs.rows || this->cols != rhs.cols) {
        throw std::invalid_argument("Matrix dimensions does not match up! 1");
    }

    Matrix<T> result(this->rows, this->cols, 0);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rhs.rows; ++i) {
        for (size_t j = 0; j < rhs.cols; ++j) {
            result.mat[i][j] = this->mat[i][j] + rhs.mat[i][j];
        }
    }
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs) {
    if(this->rows != rhs.rows || this->cols != rhs.cols) {
        throw std::invalid_argument("Matrix dimensions does not match up! 2");
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rhs.rows; ++i) {
        for (size_t j = 0; j < rhs.cols; ++j) {
            this->mat[i][j] += rhs.mat[i][j];
        }
    }
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& rhs) {
    if(this->rows != rhs.rows || this->cols != rhs.cols) {
        throw std::invalid_argument("Matrix dimensions does not match up! 3");
    }

    Matrix<T> result(this->rows, this->cols, 0);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rhs.rows; ++i) {
        for (size_t j = 0; j < rhs.cols; ++j) {
            result.mat[i][j] = this->mat[i][j] - rhs.mat[i][j];
        }
    }
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& rhs) {
    if(this->rows != rhs.rows || this->cols != rhs.cols) {
        throw std::invalid_argument("Matrix dimensions does not match up! 4");
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rhs.rows; ++i) {
        for (size_t j = 0; j < rhs.cols; ++j) {
            this->mat[i][j] -= rhs.mat[i][j];
        }
    }
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& rhs) const {
    if(this->rows != rhs.rows || this->cols != rhs.cols) {
        throw std::invalid_argument("Matrix dimensions does not match up! 5");
    }

    Matrix<T> result(this->rows, this->cols, 0);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rhs.rows; ++i) {
        for (size_t j = 0; j < rhs.cols; ++j) {
            result.mat[i][j] = this->mat[i][j] * rhs.mat[i][j];
        }
    }
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& rhs) {
    if(this->rows != rhs.rows || this->cols != rhs.cols) {
        throw std::invalid_argument("Matrix dimensions does not match up! 6");
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rhs.rows; ++i) {
        for (size_t j = 0; j < rhs.cols; ++j) {
            this->mat[i][j] *= rhs.mat[i][j];
        }
    }
    return *this;
}

// Scalar addition operator
template <typename T>
Matrix<T> Matrix<T>::operator+(const T& scalar) {
    Matrix<T> result(this->rows, this->cols, 0);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            result.mat[i][j] = this->mat[i][j] + scalar;
        }
    }
    return result;
}

// Scalar subtraction operator
template <typename T>
Matrix<T> Matrix<T>::operator-(const T& scalar) {
    Matrix<T> result(this->rows, this->cols, 0);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            result.mat[i][j] = this->mat[i][j] - scalar;
        }
    }
    return result;
}

// Scalar multiplication operator
template <typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar) {
    Matrix<T> result(this->rows, this->cols, 0);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            result.mat[i][j] = this->mat[i][j] * scalar;
        }
    }
    return result;
}

// Scalar division operator
template <typename T>
Matrix<T> Matrix<T>::operator/(const T& scalar) {
    if(scalar == 0) {
        throw std::invalid_argument("Divison by zero!");
    }

    Matrix<T> result(this->rows, this->cols, 0);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            result.mat[i][j] = this->mat[i][j] / scalar;
        }
    }
    return result;
}

// Matrix-vector multiplication operator
template <typename T>
std::vector<T> Matrix<T>::operator*(const std::vector<T> &rhs) {
    if (this->cols != rhs.size()) {
        throw std::invalid_argument("Matrix columns and vector size must match up!");
    }

    std::vector<T> result(this->rows, 0);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            result[i] += this->mat[i][j] * rhs[j];
        }
    }
    return result;
}

// Function to return the diagonal elements of the matrix as a vector
template <typename T>
std::vector<T> Matrix<T>::diag_vec() {
    size_t length = std::min(this->rows, this->cols);
    std::vector<T> result(length);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < length; ++i) {
        result[i] = this->mat[i][i];
    }
    return result;
}

// Element access/modification operator
template <typename T>
T& Matrix<T>::operator()(const unsigned& row, const unsigned& col) {
    if (row >= this->rows || col >= this->cols) {
        throw std::out_of_range("Matrix indices are out of range!");
    }
    return this->mat[row][col];
}

// Getter for the number of rows
template <typename T>
unsigned Matrix<T>::get_rows() const {
    return this->rows;
}

// Getter for the number of columns
template <typename T>
unsigned Matrix<T>::get_cols() const {
    return this->cols;
}

// Transpose function
template <typename T>
Matrix<T> Matrix<T>::transpose() {
    Matrix<T> result(this->cols, this->rows, 0);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            result.mat[j][i] = this->mat[i][j];
        }
    }
    return result;
}

// Dot-product function
template <typename T>
Matrix<T> Matrix<T>::dotNoTiling(const Matrix<T>& rhs) const {
    if (this->cols != rhs.rows) {
        throw std::invalid_argument("The columns of Matrix A is not equal to the rows of Matrix B");
    }

    Matrix<T> result(this->rows, rhs.cols, 0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < this->rows; ++i) {
        for (int k = 0; k < this->cols; ++k) {
            T temp = this->mat[i][k];
            for (int j = 0; j < rhs.cols; ++j) {
                result(i, j) += temp * rhs.mat[k][j];
            }
        }
    }
    return result;
}

// Dot-product function
template <typename T>
Matrix<T> Matrix<T>::dotTiling(const Matrix<T>& rhs) const {
    unsigned tileSize = 128;
    if (this->cols != rhs.rows) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }

    Matrix<T> result(this->rows, rhs.cols, 0);

    for (int i = 0; i < this->rows; i += tileSize) {
        for (int k = 0; k < this->cols; k += tileSize) {
            for (int j = 0; j < rhs.cols; j += tileSize) {

                for (int ii = i; ii < std::min<int>(i + tileSize, this->rows); ++ii) {
                    for (int kk = k; kk < std::min<int>(k + tileSize, this->cols); ++kk) {
                        T temp = this->mat[ii][kk];

                        for (int jj = j; jj < std::min<int>(j + tileSize, rhs.cols); ++jj) {
                            result.mat[ii][jj] += temp * rhs.mat[kk][jj];
                        }
                    }
                }

            }
        }
    }
    return result;
}

// Hadamard product
template <typename T>
Matrix<T> Matrix<T>::Hadamard(const Matrix<T>& rhs) const {
    if (this->rows != rhs.rows || this->cols != rhs.cols) {
        throw std::invalid_argument("Matrices does not have the same dimensions");
    }

    Matrix<T> result(this->rows, this->cols, 0);

    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            result(i, j) = this->mat[i][j] * rhs.mat[i][j];
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::sum(int dim) const {
    if (dim == 0) { // Sum across columns
        Matrix<T> result(1, cols, T(0));
        for (size_t j = 0; j < cols; ++j) {
            for (size_t i = 0; i < rows; ++i) {
                result(0, j) += mat[i][j];
            }
        }
        return result;
    } else if (dim == 1) { // Sum across rows
        Matrix<T> result(rows, 1, T(0));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, 0) += mat[i][j];
            }
        }
        return result;
    } else {
        throw std::invalid_argument("Invalid dimension for sum. Use 0 for columns or 1 for rows.");
    }
}

// Print function
template <typename T>
void Matrix<T>::print() const {
    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            std::cout << this->mat[i][j] << " ";
        }
        std::cout << std::endl; // New line at the end of each row
    }
}
