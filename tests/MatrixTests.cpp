#include "../include/Matrix.h"

#include <vector>
#include <iostream>
#include <assert.h>
#include <sstream>
#include <string>
#include <chrono>

void testDefaultConstructor() {
    Matrix<int> mat;
    assert(mat.get_rows() == 0);
    assert(mat.get_cols() == 0);

    std::cout << "Default Constructor Test Passed." << std::endl;
}

void testParameterizedConstructor() {
    Matrix<int> mat(2, 5, 5);
    assert(mat.get_rows() == 2);
    assert(mat.get_cols() == 5);

    for (int i = 0; i < mat.get_rows(); ++i) {
        for (int j = 0; j < mat.get_cols(); ++j) {
            assert(mat(i, j) == 5);
        }
    }

    std::cout << "Parameterized Constructor Test Passed." << std::endl;
}

void testCopyConstructor() {
    Matrix<int> mat1(2, 5, 5);
    Matrix<int> mat2 = mat1;

    assert(mat2.get_rows() == mat1.get_rows());
    assert(mat2.get_cols() == mat1.get_cols());

    for (int i = 0; i < mat2.get_rows(); ++i) {
        for (int j = 0; j < mat2.get_cols(); ++j) {
            assert(mat2(i, j) == mat1(i, j));
        }
    }
    std::cout << "Copy Constructor Test Passed." << std::endl;
}

void testCopyAssignmentOperator() {
    Matrix<int> mat1(2, 2, 4);
    Matrix<int> mat2;
    mat2 = mat1; // Using copy assignment operator
    assert(mat2.get_rows() == mat1.get_rows());
    assert(mat2.get_cols() == mat1.get_cols());
    for (unsigned i = 0; i < mat2.get_rows(); ++i) {
        for (unsigned j = 0; j < mat2.get_cols(); ++j) {
            assert(mat2(i, j) == mat1(i, j));
        }
    }
    std::cout << "Copy Assignment Operator Test Passed." << std::endl;
}

void testAdditionOperator() {
    Matrix<int> mat1(2, 2, 1);
    Matrix<int> mat2(2, 2, 2);
    Matrix<int> result = mat1 + mat2;

    for (int i = 0; i < result.get_rows(); ++i) {
        for (int j = 0; j < result.get_cols(); ++j) {
            assert(result(i, j) == 3);
        }
    }
    std::cout << "Addition Operator Test Passed." << std::endl;
}

void testSubtractionOperator() {
    Matrix<int> mat1(2, 2, 5);
    Matrix<int> mat2(2, 2, 2);
    Matrix<int> result = mat1 - mat2;

    for (int i = 0; i < result.get_rows(); ++i) {
        for (int j = 0; j < result.get_cols(); ++j) {
            assert(result(i, j) == 3);
        }
    }
    std::cout << "Subtraction Operator Test Passed." << std::endl;
}

void testMultiplicationOperator() {
    Matrix<int> mat1(2, 2, 5);
    Matrix<int> mat2(2, 2, 2);
    Matrix<int> result = mat1 * mat2;

    for (int i = 0; i < result.get_rows(); ++i) {
        for (int j = 0; j < result.get_cols(); ++j) {
            assert(result(i, j) == 10);
        }
    }
    std::cout << "Multiplication Operator Test Passed." << std::endl;
}

void testScalarOperations() {
    Matrix<int> mat(2, 2, 2);

    // Test scalar addition
    Matrix<int> addResult = mat + 3;
    for (unsigned i = 0; i < addResult.get_rows(); ++i) {
        for (unsigned j = 0; j < addResult.get_cols(); ++j) {
            assert(addResult(i, j) == 5); // 2 + 3
        }
    }

    // Test scalar subtraction
    Matrix<int> subResult = mat - 1;
    for (unsigned i = 0; i < subResult.get_rows(); ++i) {
        for (unsigned j = 0; j < subResult.get_cols(); ++j) {
            assert(subResult(i, j) == 1); // 2 - 1
        }
    }

    // Test scalar multiplication
    Matrix<int> mulResult = mat * 2;
    for (unsigned i = 0; i < mulResult.get_rows(); ++i) {
        for (unsigned j = 0; j < mulResult.get_cols(); ++j) {
            assert(mulResult(i, j) == 4); // 2 * 2
        }
    }

    // Test scalar division
    Matrix<int> divResult = mat / 2;
    for (unsigned i = 0; i < divResult.get_rows(); ++i) {
        for (unsigned j = 0; j < divResult.get_cols(); ++j) {
            assert(divResult(i, j) == 1); // 2 / 2
        }
    }

    std::cout << "Scalar Operations Tests Passed." << std::endl;
}

void testMatrixVectorMultiplication() {
    Matrix<int> mat(2,2,1);
    std::vector<int> vec = {2, 3};
    std::vector<int> result = mat * vec;
    assert(result.size() == 2);
    assert(result[0] == 5);
    assert(result[1] == 5);
    std::cout << "Matrix-Vector Multiplication Test Passed." << std::endl;
}

void testTranspose() {
    // Create a 2x3 matrix with specific values
    Matrix<int> mat(2, 3);
    mat(0, 0) = 1; mat(0, 1) = 2; mat(0, 2) = 3;
    mat(1, 0) = 4; mat(1, 1) = 5; mat(1, 2) = 6;

    // Perform the transpose operation
    Matrix<int> transposed = mat.transpose();

    // Check the dimensions of the transposed matrix
    assert(transposed.get_rows() == 3);
    assert(transposed.get_cols() == 2);

    // Check the values of the transposed matrix
    assert(transposed(0, 0) == 1); assert(transposed(1, 0) == 2); assert(transposed(2, 0) == 3);
    assert(transposed(0, 1) == 4); assert(transposed(1, 1) == 5); assert(transposed(2, 1) == 6);

    std::cout << "Transpose Test Passed." << std::endl;
}

void testPrintFunction() {
    Matrix<int> mat(2, 2, 3);

    // Backup the original buffer
    std::streambuf* originalBuffer = std::cout.rdbuf();

    // Create a stringstream buffer to capture the output
    std::stringstream buffer;
    std::cout.rdbuf(buffer.rdbuf());

    // Call the print function
    mat.print();

    // Restore the original buffer
    std::cout.rdbuf(originalBuffer);

    // Define the expected output
    std::string expectedOutput = "3 3 \n3 3 \n";

    // Check if the captured output matches the expected output
    assert(buffer.str() == expectedOutput);

    std::cout << "Print Function Test Passed." << std::endl;
}

void testDiagonalExtraction() {
    // Create a 3x3 matrix with specific values
    Matrix<int> mat(3, 3);
    mat(0, 0) = 1; mat(0, 1) = 2; mat(0, 2) = 3;
    mat(1, 0) = 4; mat(1, 1) = 5; mat(1, 2) = 6;
    mat(2, 0) = 7; mat(2, 1) = 8; mat(2, 2) = 9;

    // Perform diagonal extraction
    std::vector<int> diag = mat.diag_vec();

    // Check the size of the diagonal vector
    assert(diag.size() == 3);

    // Check the values of the diagonal vector
    assert(diag[0] == 1);
    assert(diag[1] == 5);
    assert(diag[2] == 9);

    std::cout << "Diagonal Extraction Test Passed." << std::endl;
}

void testMatrixMatrixMultiplication() {
    // Define two matrices
    Matrix<int> mat1(2, 3);
    mat1(0, 0) = 1; mat1(0, 1) = 2; mat1(0, 2) = 3;
    mat1(1, 0) = 4; mat1(1, 1) = 5; mat1(1, 2) = 6;

    Matrix<int> mat2(3, 2);
    mat2(0, 0) = 7; mat2(0, 1) = 8;
    mat2(1, 0) = 9; mat2(1, 1) = 10;
    mat2(2, 0) = 11; mat2(2, 1) = 12;

    // Perform matrix-matrix multiplication
    Matrix<int> result = mat1.dotNoTiling(mat2);

    // Define the expected result of the multiplication
    Matrix<int> expected(2, 2);
    expected(0, 0) = 58; expected(0, 1) = 64;
    expected(1, 0) = 139; expected(1, 1) = 154;

    // Check if the result matches the expected matrix
    for (unsigned i = 0; i < result.get_rows(); ++i) {
        for (unsigned j = 0; j < result.get_cols(); ++j) {
            assert(result(i, j) == expected(i, j));
        }
    }
    std::cout << "Matrix-Matrix Multiplication Test Passed." << std::endl;
}

void testDotProductPerformanceNoTiling() {
    // Define the size of the matrices
    const int numRows = 500; // Adjust as needed
    const int numCols = 5000; // Adjust as needed

    // Initialize two matrices with random or predetermined values
    Matrix<double> matA(numRows, numCols); // Fill with values
    Matrix<double> matB(numCols, numRows); // Fill with values

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the dot product
    Matrix<double> result = matA.dotNoTiling(matB);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and print the elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Dot Product Performance: " << elapsed.count() << " seconds." << std::endl;
}

void testDotProductPerformanceTiling() {
    // Define the size of the matrices
    const int numRows = 500; // Adjust as needed
    const int numCols = 5000; // Adjust as needed

    // Initialize two matrices with random or predetermined values
    Matrix<double> matA(numRows, numCols); // Fill with values
    Matrix<double> matB(numCols, numRows); // Fill with values

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the dot product
    Matrix<double> result = matA.dotTiling(matB);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and print the elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Dot Product Performance: " << elapsed.count() << " seconds." << std::endl;
}

void testHadamardProduct() {
    // Create and fill two matrices
    Matrix<int> mat1(2, 2, 1);  // A 2x2 matrix filled with 1s
    Matrix<int> mat2(2, 2);     // Another 2x2 matrix
    mat2(0, 0) = 2; mat2(0, 1) = 3;
    mat2(1, 0) = 4; mat2(1, 1) = 5;

    // Calculate the Hadamard product
    Matrix<int> result = mat1.Hadamard(mat2);

    // Define the expected result
    Matrix<int> expected(2, 2);
    expected(0, 0) = 2; expected(0, 1) = 3;
    expected(1, 0) = 4; expected(1, 1) = 5;

    // Assert that the result matches the expected output
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            assert(result(i, j) == expected(i, j));
        }
    }

    std::cout << "Hadamard Product Test Passed." << std::endl;
}

/*
int main() {
    testDefaultConstructor();
    testParameterizedConstructor();
    testCopyConstructor();
    testCopyAssignmentOperator();
    testAdditionOperator();
    testSubtractionOperator();
    testMultiplicationOperator();
    testScalarOperations();
    testMatrixVectorMultiplication();
    testTranspose();
    testPrintFunction();
    testDiagonalExtraction();
    testMatrixMatrixMultiplication();
    testHadamardProduct();

    testDotProductPerformanceNoTiling();
    testDotProductPerformanceTiling();
    return 0;
}*/
