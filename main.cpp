//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/Bitmap.h"
#include "Utils/Utils.h"
#include "Network/Network.h"
#include <opencv2/opencv.hpp>
int main(){
    cv::Mat mat = cv::imread("resources/aPhoto.jpg");
    cn::Network network(800, 800, 3);

    int size = mat.cols * mat.rows * mat.channels();

  //  byte converted [size] = mat.

    cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);

   // bitmap.setCell(100, 100, 0, 255);
   // bitmap.setCell(100, 100, 1, 255);
   // bitmap.setCell(100, 100, 2, 255);

    std::vector<cn::byte> decodedRaw (bitmap.w * bitmap.h * bitmap.d);
    cn::Utils::convert(bitmap.data(), decodedRaw.data(), bitmap.w, bitmap.h, bitmap.d, 0, 1);

    cv::Mat decoded = cv::Mat(bitmap.h, bitmap.w, CV_8UC(bitmap.d), decodedRaw.data());

  //  network.appendConvolutionLayer(3, 3, 3, 1);
  //  network.appendConvolutionLayer(3, 3, 1, 1);

    //network.feed()
    cv::imshow("image", decoded);
    cv::waitKey(10000);


//
 //   std::cout<<b;
    return 0;
}