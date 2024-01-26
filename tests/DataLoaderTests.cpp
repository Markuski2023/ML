#include "../include/Matrix.h"
#include "../include/DataLoader.h"

// Test for DataLoader constructor
void test_DataLoaderConstructor() {
    try {
        DataLoader loader;
        std::cout << "Constructor test passed." << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Constructor test failed: " << e.what() << std::endl;
    }
}

// Test for DataLoader loadCSV with valid file
void test_loadCSV_ValidFile() {
    try {
        DataLoader loader;
        loader.loadCSV("mock_data.csv", 16, true);
        std::cout << "loadCSV test passed for valid file." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "loadCSV test failed for valid file: " << e.what() << std::endl;
    }
}


// Test for DataLoader loadCSV correctness
void test_loadCSV_Correctness() {

}

void test_print() {
    DataLoader loader;
    loader.loadCSV("mock_data.csv", 2, true);
    loader.print();
}

// Main function to run all tests
// int main() {
//     test_DataLoaderConstructor();
//     test_loadCSV_ValidFile();
//     test_loadCSV_Correctness();
//     test_print();
//
//     return 0;
// }