#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <utility> // For std::pair

// Assuming Matrix is a custom class for matrix operations
#include "Matrix.h"

template <typename T>
class DataLoader {
public:
    // Constructor to initialize the DataLoader with a file path
    DataLoader(const std::string& filePath);

    // Function to load data from the file
    void loadCSV();

    // Function to preprocess the loaded data
    void preprocess();

    // Function to get the input size (number of features)
    int getInputSize() const;

    // Function to get the output size (number of classes or output dimensions)
    int getOutputSize() const;

    // Function to split the dataset into training and test sets
    void splitData(double trainSplitRatio);

    // Function to get training and test data batches
    std::pair<std::vector<Matrix<T>>, std::vector<Matrix<T>>> getBatches(int batchSize) const;

private:
    std::string filePath;
    std::vector<Matrix<T>> inputData;
    std::vector<Matrix<T>> outputData;

};

#endif // DATA_LOADER_H
#include "../src/DataLoader.tpp"