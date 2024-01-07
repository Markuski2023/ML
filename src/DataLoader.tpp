#include "../include/DataLoader.h"

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>

template <typename T>
DataLoader<T>::DataLoader() {}

template<typename T>
void DataLoader<T>::loadCSV(const std::string& filePath, T batch_size, bool shuffle) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("File cannot be opened or does not exist: " + filePath);
    }
    std::string line;
    std::vector<std::vector<T>> data, batch;

    // Read data from file
    while (getline(file, line)) {
        std::vector<T> row;
        std::stringstream ss(line);
        T value;

        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }

        data.push_back(row);
    }
    file.close();

    if (shuffle) {
        std::random_shuffle(data.begin(), data.end());
    }

    // Create batches
    for (size_t i = 0; i < data.size(); i++) {
        batch.push_back(data[i]);
        if (batch.size() == batch_size || i == data.size() - 1) {
            Matrix<T> matrix(batch);
            inputData.push_back(matrix);
            batch.clear();
        }
    }
}


template<typename T>
void DataLoader<T>::print() {
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