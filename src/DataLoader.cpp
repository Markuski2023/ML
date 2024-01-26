#include "../include/DataLoader.h"

#include <stdexcept>
#include <fstream>
#include <sstream>


DataLoader::DataLoader() {}

void DataLoader::loadCSV(const std::string& filePath, double batch_size, bool shuffle) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("File cannot be opened or does not exist: " + filePath);
    }
    std::string line;
    std::vector<std::vector<double>> data, batch;

    // Read data from file
    while (getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        double value;

        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        data.push_back(row);
    }
    file.close();

    // Create batches
    for (size_t i = 0; i < data.size(); i++) {
        batch.push_back(data[i]);
        if (batch.size() == batch_size || i == data.size() - 1) {
            Matrix<double> matrix(batch);
            inputData.push_back(matrix);
            batch.clear();
        }
    }
}

void DataLoader::print() {
    for (size_t i = 0; i < inputData.size(); ++i) {
        std::cout << "Batch " << i + 1 << ":" << std::endl;
        for (size_t row = 0; row < inputData[i].get_rows(); ++row) {
            for (size_t column = 0; column < inputData[i].get_cols(); ++column) {
                std::cout << inputData[i](row, column) << " "; // Print element followed by a space
            }
            std::cout << std::endl; // New line after each row
        }
        std::cout << std::endl; // Additional new line after each Matrix
    }
}
