#include "../include/Matrix.h"
#include "../include/DataLoader.h"

// Test for DataLoader constructor
void test_DataLoaderConstructor() {
    std::cout << "Testing DataLoader Constructor..." << std::endl;

    try {
        DataLoader<double> loader("valid_file_path.csv");
        std::cout << "Constructor test passed." << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Constructor test failed: " << e.what() << std::endl;
    }
}

// Test for DataLoader loadCSV with valid file
void test_loadCSV_ValidFile() {
    std::cout << "Testing loadCSV with valid file..." << std::endl;

    try {
        DataLoader<double> loader("valid_file_path.csv");
        loader.loadCSV();
        std::cout << "loadCSV test passed for valid file." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "loadCSV test failed for valid file: " << e.what() << std::endl;
    }
}

// Test for DataLoader loadCSV with invalid file
void test_loadCSV_InvalidFile() {
    std::cout << "Testing loadCSV with invalid file..." << std::endl;

    try {
        DataLoader<double> loader("invalid_file_path.csv");
        loader.loadCSV();
        std::cerr << "loadCSV test failed for invalid file: No exception thrown." << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "loadCSV test passed for invalid file: " << e.what() << std::endl;
    }
}

// Test for DataLoader loadCSV correctness
void test_loadCSV_Correctness() {
    std::cout << "Testing loadCSV correctness..." << std::endl;
}

// Main function to run all tests
int main() {
    test_DataLoaderConstructor();
    test_loadCSV_ValidFile();
    test_loadCSV_InvalidFile();
    test_loadCSV_Correctness();

    return 0;
}
