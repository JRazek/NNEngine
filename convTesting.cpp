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
//    cn::Bitmap<cn::byte> bitmap = cn::Utils::normalize(cn::Bitmap<cn::byte>(mat.cols, mat.rows, mat.dims, mat.data, 1));
    cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data);

    //network.appendConvolutionLayer(3,3,3,1);

    //network.feed(bitmap);

    cn::Bitmap<cn::byte> resampled = cn::Utils::resize(bitmap, 1000, 1000);

    cn::byte dataStorage [resampled.w * resampled.h * resampled.d];

    cv::Mat resampledImg(1000, 1000, CV_8UC(3u), dataStorage);


    cn::Utils::convert(resampled.data(), dataStorage, resampled.w, resampled.h, resampled.d, 0, 1);

    cv::imshow("img", resampledImg);

    cv::waitKey(10000);
    //std::pair<int, int> neighbor = quadTree.getNearestNeighbour({4, 4});
    return 0;
}