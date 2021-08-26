//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/Bitmap.h"
#include "Utils/Utils.h"
#include "Network/Network.h"
#include "Network/layers/ConvolutionLayer.h"
#include "Network/layers/FlatteningLayer.h"
#include "LearningModels/Backpropagation.h"
#include <opencv2/opencv.hpp>
int main(){
    cv::Mat mat = cv::imread("resources/aPhoto.jpg");
    cn::Network network(33, 33, 3, 1);

    cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);
    bitmap = cn::Utils::resize<cn::byte>(bitmap, 100, 1);
    ReLU reLu;
    Sigmoid sigmoid;

    cn::Backpropagation backpropagation(network, 0.01);

    const int outputSize = 10;
    network.appendFlatteningLayer();
    network.appendFFLayer(10, sigmoid);
    network.appendFFLayer(10, sigmoid);
    network.appendFFLayer(outputSize, sigmoid);
    network.initRandom();

    network.ready();

    cn::Bitmap<float> target (outputSize, 1, 1);
    for(int i = 0; i < outputSize; i ++){
        target.setCell(i, 0, 0, 0.5);
    }

    for(int i = 0; i < 100000; i ++) {
        network.feed(bitmap);
        backpropagation.propagate(target);
    }


    return 0;
}