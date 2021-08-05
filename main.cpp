//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/Bitmap.h"
#include "Network/Network.h"
#include <opencv2/opencv.hpp>
int main(){
    cv::Mat mat = cv::imread("resources/aPhoto.jpg");
    cn::Network network(800, 800, 3);

    int size = mat.cols * mat.rows * mat.channels();
    cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels());

    std::copy(mat.data, mat.data + size, bitmap.data());


    cv::Mat decoded = cv::Mat(bitmap.h, bitmap.w, CV_8UC(bitmap.d), bitmap.data()).clone();
    network.appendConvolutionLayer(3, 3, 3, 1);
    network.appendConvolutionLayer(3, 3, 1, 1);

    //network.feed()
    //cv::imshow("image", decoded);
   // cv::waitKey(10000);

   // network.appendLayer()

//
 //   std::cout<<b;
    return 0;
}