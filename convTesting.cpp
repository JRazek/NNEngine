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
    cn::Network network(2000, 3000, 3);

    cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);

    network.appendConvolutionLayer(3,3,3,1);

    cn::Bitmap<cn::byte> resampled = cn::Utils::resize(bitmap, 3000, 4500);

    auto * dataStorage = new cn::byte [resampled.w * resampled.h * resampled.d];

    cn::Utils::convert(resampled.data(), dataStorage, resampled.w, resampled.h, resampled.d, 0, 1);


    cv::Mat resampledImg(resampled.h, resampled.w, CV_8UC(resampled.d), dataStorage);

    cn::Bitmap<cn::byte> transformed = cn::Utils::transform(resampled, {1, 0,0,1});

    cv::imshow("img", resampledImg);


    while(cv::waitKey() != 48);

    //std::pair<int, int> neighbor = quadTree.getNearestNeighbour({4, 4});

    //delete [] dataStorage;
    return 0;
}