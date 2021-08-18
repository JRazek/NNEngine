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

    cn::Bitmap<cn::byte> resampled = cn::Utils::resize(bitmap, 1000, 1500);

    cn::Bitmap<cn::byte> rotated = cn::Utils::rotate(resampled, M_PI/2.f);

    auto * dataStorage = new cn::byte [rotated.w * rotated.h * rotated.d];

    cn::Utils::convert(rotated.data(), dataStorage, rotated.w, rotated.h, rotated.d, 0, 1);

    cv::Mat transformedImg(rotated.h, rotated.w, CV_8UC(rotated.d), dataStorage);

    cv::imshow("img", transformedImg);


    while(cv::waitKey() != 48);

    //std::pair<int, int> neighbor = quadTree.getNearestNeighbour({4, 4});

    delete [] dataStorage;
    return 0;
}