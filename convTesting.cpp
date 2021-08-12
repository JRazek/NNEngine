//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/Bitmap.h"
#include "Utils/Utils.h"
#include "Network/Network.h"
#include "Network/layers/ConvolutionLayer.h"
#include <opencv2/opencv.hpp>

int main(){
    cv::Mat mat = cv::imread("resources/aPhoto.jpg");
    cn::Network network(800, 800, 3);

    //int w, int h, int d, const T* data, int inputType = 0
    cn::Bitmap<float> bitmap = cn::Utils::normalize(cn::Bitmap<cn::byte>(mat.cols, mat.rows, mat.dims, mat.data, 1));
    network.appendConvolutionLayer(3,3,3,1);

    network.feed(bitmap);

    //std::pair<int, int> neighbor = quadTree.getNearestNeighbour({4, 4});
    return 0;
}