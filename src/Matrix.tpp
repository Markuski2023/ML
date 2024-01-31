#include "../include/Matrix.h"

#include <omp.h>
#include <stdexcept>
#include <random>
#include <chrono>
#include <map>

// Default constructor
template <typename T>
Matrix<T>::Matrix() : rows(0), cols(0) {}

// Constructor for vector of vector
template <typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>> input) {
    if (input.empty()) {
        throw std::invalid_argument("Constructor error: Input vector of vectors is empty.");
    }

    if (input[0].empty()) {
        throw std::invalid_argument("First row of input vector is empty");
    }

    rows = input.size();
    cols = input[0].size();

    mat.resize(rows, std::vector<T>(cols));

    for (size_t i = 0; i < rows; ++i) {
        if (input[i].size() != cols) {
            throw std::invalid_argument("Constructor error: Inconsistent number of columns in input vector.");
        }
        for (size_t j = 0; j < cols; ++j) {
            mat[i][j] = input[i][j];
        }
    }
}

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

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& rhs) {
    // Self-assignment check to avoid unnecessary work
    if (this == &rhs) {
        return *this;
    }

    // Copy the size
    rows = rhs.rows;
    cols = rhs.cols;

    // Copy the data
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
Matrix<T> Matrix<T>::dotTiling(const Matrix<T>& rhs) const {
    if (this->cols != rhs.rows) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }

    unsigned tileSize = 128; // Tiling size, should be a power of 2 for best cache performance
    Matrix<T> result(this->rows, rhs.cols, 0);

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < this->rows; i += tileSize) {
        for (int k = 0; k < this->cols; k += tileSize) {
            for (int j = 0; j < rhs.cols; j += tileSize) {

                Matrix<T> subMat1(this, i, i + tileSize - 1, k, k + tileSize - 1);
                Matrix<T> subMat2(rhs, k, k + tileSize - 1, j, j + tileSize - 1);

                for (int ii = 0; ii < subMat1.rows; ++ii) {
                    for (int jj = 0; jj < subMat2.cols; ++jj) {

                        T temp = subMat1.mat[ii][0] * subMat2.mat[0][jj];

                        for (int kk = 1; kk < subMat2.rows; ++kk) {
                            temp += subMat1.mat[ii][kk] * subMat2.mat[kk][jj];
                        }

                        result.mat[i + ii][j + jj] += temp;
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

    Matrix<T> result(*this); // Copy input matrix to avoid modifying original

    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            result.mat[i][j] *= rhs.mat[i][j]; // Multiply corresponding elements in place
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