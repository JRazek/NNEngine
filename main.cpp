//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/Bitmap.h"
#include "Utils/Utils.h"
#include "Utils/dataStructures/QuadTree.h"
#include "Network/Network.h"
#include "Network/layers/ConvolutionLayer.h"
#include <opencv2/opencv.hpp>
int main(){
    cv::Mat mat = cv::imread("resources/aPhoto.jpg");
    cn::Network network(800, 800, 3);

    cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);

//    bitmap.setCell(200, 100, 0, 255);
//    bitmap.setCell(200, 100, 1, 255);
//    bitmap.setCell(200, 100, 2, 255);


    float kernelData [27];
    for(int i = 0; i < sizeof(kernelData)/sizeof(float); i ++){
        kernelData[i] = (float)(i % 9) + 1;
    }

    float sampleImageData [75];
    std::fill(sampleImageData, sampleImageData + 75, 1);

    cn::Bitmap<float> kernel(3, 3, 3, kernelData);
    cn::Bitmap<float> sampleImage(5, 5, 3, sampleImageData);


    cn::Bitmap<float> result = cn::Utils::convolve(kernel, sampleImage,1, 1);

    std::vector<float> encodedRGB(result.w * result.h * result.d);
    cn::Utils::convert<float>(result.data(), encodedRGB.data(), result.w, result.h, result.d, 0, 1);

    cv::Mat decoded = cv::Mat(result.h, result.w, CV_8UC(result.d), encodedRGB.data());


    QuadTree quadTree(1024, 1024);
    int t = 3;
    PointData pointData1({23, 32}, &t);
    quadTree.insertPoint(pointData1);

    //std::pair<int, int> neighbor = quadTree.getNearestNeighbour({4, 4});
    return 0;
}