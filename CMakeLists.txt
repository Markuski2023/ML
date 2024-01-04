cmake_minimum_required(VERSION 3.27)
project(ML)

set(CMAKE_CXX_STANDARD 17)

add_executable(ML main.cpp
        src/Matrix.tpp
        include/Matrix.h
        include/Optimizers/Optimizer.h
        src/NeuralNetwork.tpp
        include/NeuralNetwork.h
        src/DenseLayer.tpp
        include/Layers/DenseLayer.h
        include/Layers/Layer.h
        tests/MatrixTests.cpp
        tests/NeuralTests.cpp
        include/Optimizers/SGD.h
        include/Layers/ActivationLayer.h
        include/Layers/ReLU.h
        include/ErrorFunctions/Error.h
        include/ErrorFunctions/MeanSquaredError.h
        include/ErrorFunctions/BinaryCrossEntropyError.h
        include/Layers/Sigmoid.h
        include/Layers/Softmax.h
        include/ErrorFunctions/CategoricalCrossEntropyError.h
)