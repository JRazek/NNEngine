//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Network/Network.h"
#include <opencv2/opencv.hpp>
int main(){
    cv::Mat mat = cv::imread("resources/aPhoto.jpg");
    Network network;

    int size = mat.cols * mat.rows * mat.channels();

    Bitmap bitmap(mat.cols, mat.rows, mat.channels());

    std::copy(mat.data, mat.data + size, bitmap.getData());

    byte b = bitmap.getByte(99, 99, 2);

    cv::Mat decoded = cv::Mat(bitmap.h, bitmap.w, CV_8UC(bitmap.d), bitmap.getData()).clone();

    cv::imshow("image", decoded);
    cv::waitKey(10000);


    std::cout<<b;
    return 0;
}