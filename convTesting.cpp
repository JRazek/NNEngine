//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/dataStructures/Bitmap.h"
#include "Utils/Utils.h"
#include "Network/Network.h"
#include "LearningModels/Backpropagation.h"
#include <opencv2/opencv.hpp>
#include "Utils/dataStructures/PrefixSum2DArr.h"

int main(){
    cv::Mat mat = cv::imread("resources/aPhoto.jpg");
    cn::Network network(100, 100, 3, 1);

    cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);
    ReLU reLu;
    Sigmoid sigmoid;

    cn::Backpropagation backpropagation(network, 0.1);

    const int outputSize = 10;
    network.appendConvolutionLayer(3, 3, 1, reLu, 0, 0, 4, 4);
    network.appendMaxPoolingLayer(2, 2);
    network.appendConvolutionLayer(3, 3, 1, reLu, 0, 0, 2, 2);
    network.appendConvolutionLayer(3, 3, 1, reLu, 0, 0, 2, 2);
    network.appendBatchNormalizationLayer();
    network.appendFlatteningLayer();
    network.appendFFLayer(outputSize, sigmoid);
    network.initRandom();

    network.ready();

    cn::Bitmap<float> target (outputSize, 1, 1);
    for(int i = 0; i < outputSize; i ++){
        target.setCell(i, 0, 0, 0.5);
    }

    for(int i = 0; i < 1; i ++) {
        network.feed(bitmap);
        std::cout<<i<<": "<<backpropagation.getError(target)<<"\n";
        backpropagation.propagate(target);
    }

    PrefixSum2D<long long> prefixSum2D(bitmap);
    return 0;
}