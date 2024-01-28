#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <utility> // For std::pair

// Assuming Matrix is a custom class for matrix operations
#include "Matrix.h"

class DataLoader {
public:
    // Constructor to initialize the DataLoader with a file path
    DataLoader();

    // Function to load data from the file
    void loadCSV(const std::string& filePath, int batch_size=1, bool shuffle=true);

    // Function to print out content of loader
    void print();

    // Function to preprocess the loaded data
    // void preprocess();

    // Function to split the dataset into training and test sets
    void splitData(double trainSplitRatio);

    // Function to get training and test data batches
    std::pair<std::vector<Matrix<double>>, std::vector<Matrix<double>>> getBatches(int batchSize) const;

private:
    std::string filePath;
    std::vector<Matrix<double>> inputData;
    std::vector<std::vector<double>> outputData;
};

#endif // DATA_LOADER_H