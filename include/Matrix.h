#ifndef ML_MATRIX_H
#define ML_MATRIX_H

#include <vector>
#include <iostream>
#include <random>

#include "../include/Matrix.h"


// Templated class to allow for flexibility with different data types
template <typename T>
class Matrix {
public:
    // Default constructor
    Matrix();

    // Constructor for vector of vectors
    Matrix(std::vector<std::vector<T>>);

    // Parameterized constructor for creating a matrix with given dimensions and initial value
    Matrix(unsigned rows, unsigned cols, const T& initial);

    // Constructor for creating a matrix with given dimensions and random values
    Matrix(unsigned rows, unsigned cols);

    // Copy constructor for creating a copy of an existing Matrix
    Matrix(const Matrix<T> &rhs);

    // Assignment operator for assigning one matrix to another
    Matrix<T>& operator=(const Matrix<T> &rhs);

    // Addition operator overloading for adding two matrices
    Matrix<T> operator+(const Matrix<T> &rhs);
    // Addition assignment operator for adding another matrix to this one
    Matrix<T>& operator+=(const Matrix<T> &rhs);

    // Subtraction operator overloading for adding two matrices
    Matrix<T> operator-(const Matrix<T> &rhs);
    // Subtraction assignment operator for adding another matrix to this one
    Matrix<T>& operator-=(const Matrix<T> &rhs);

    // Multiplication operator overloading for adding two matrices
    Matrix<T> operator*(const Matrix<T> &rhs) const;
    // Multiplication assignment operator for adding another matrix to this one
    Matrix<T>& operator*=(const Matrix<T> &rhs);


    // Scalar addition operator for adding a scalar value to each element of the matrix
    Matrix<T> operator+(const T& scalar);

    // Scalar subtraction operator for adding a scalar value to each element of the matrix
    Matrix<T> operator-(const T& scalar);

    // Scalar multiplication operator for adding a scalar value to each element of the matrix
    Matrix<T> operator*(const T& scalar);

    // Scalar divition operator for adding a scalar value to each element of the matrix
    Matrix<T> operator/(const T& scalar);

    // Matrix-vector multiplication for multiplying matrix with a vector
    std::vector<T> operator*(const std::vector<T> &rhs);
    //Function to return the diagonal elements of the matrix as a vector
    std::vector<T> diag_vec();


    // Function to access or modify elements of the matrix
    T& operator()(const unsigned& row, const unsigned& col);

    // Getter for the number of rows/cols in the matrix
    [[nodiscard]] int getRows() const;
    [[nodiscard]] int getCols() const;

    // Function to transpose the matrix
    Matrix<T> transpose();

    // Function to do a dot-product between two matrices
    Matrix<T> dotTiling(const Matrix<T>& rhs) const;

    Matrix<T> sum(int dim = 0) const;

    std::vector<T> convertMatrixToVector(Matrix<T>& input);
    Matrix<T> convertVectorToMatrix(std::vector<T>& input, size_t rows, size_t cols);

    // Utility function to print the matrix to console
    void print() const;

    static Matrix<T> concatenateColumns(const Matrix<T>& a, const Matrix<T>& b);
    static Matrix<T> ones(unsigned rows, unsigned cols);

// Variables to store the number of rows and columns in the matrix
private:
    // 2D vector to store the matrix data
    std::vector<std::vector<T>> mat;

    // Variables to store the number of rows and columns in the matrix
    size_t rows, cols;
};

#include "..\src\Matrix.tpp"

#endif //ML_MATRIX_H