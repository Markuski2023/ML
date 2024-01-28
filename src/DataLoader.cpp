#include "../include/DataLoader.h"

#include <stdexcept>
#include <fstream>
#include <sstream>


DataLoader::DataLoader() {}

void DataLoader::loadCSV(const std::string& filePath, int batch_size, bool shuffle) {
    std::ifstream file(filePath);

    if (!file.is_open()) throw std::runtime_error("Could not open file!");


    std::string line;
    double val;
    int i = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        while (ss >> val) {

        }
        ++i;
    }
}

void DataLoader::print() {

}
