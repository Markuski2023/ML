#include "../include/DataLoader.h"

#include <stdexcept>
#include <fstream>
#include <sstream>

template <typename T>
DataLoader<T>::DataLoader(const std::string &filePath) {
    std::ifstream file(this->filePath);
    if (!file.good()) {
        throw std::runtime_error("File cannot be opened or does not exist: " + filePath);
    }
    this->filePath = filePath;

    file.close();
}

template<typename T>
void DataLoader<T>::loadCSV() {
    std::ifstream file(this->filePath);
    if (!file.is_open()) {
        throw std::runtime_error("File cannot be opened or does not exist: " + filePath);
    }
    std::string line;

    std::vector<std::vector<T>> data;

    #pragma omp parallel for collapse(2)
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

    this->inputData = Matrix<T>(data);
}
