//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/Bitmap.h"
#include "Utils/Utils.h"
#include "Utils/dataStructures/QuadTree.h"
#include "Utils/dataStructures/KDTree.h"
#include "Network/Network.h"
#include "Network/layers/ConvolutionLayer.h"
#include <opencv2/opencv.hpp>
int main(){
    cv::Mat mat = cv::imread("resources/aPhoto.jpg");
    cn::Network network(800, 800, 3);


    QuadTree quadTree(1024, 1024);
    int t = 3;
//    PointData pointData1({7,2}, &t);
//    PointData pointData2({5,4}, &t);
//    PointData pointData3({2,3}, &t);
//    PointData pointData4({9,6}, &t);
//    PointData pointData5({8,1}, &t);
//    PointData pointData6({4,7}, &t);
//      std::vector<PointData *> points = {&pointData1, &pointData2, &pointData3, &pointData4, &pointData5, &pointData6};

    PointData pointData1({7,2}, &t);
    PointData pointData2({7,4}, &t);
    PointData pointData3({7,3}, &t);
    PointData pointData4({7,6}, &t);
    PointData pointData5({7,1}, &t);
    PointData pointData6({7,7}, &t);
    std::vector<PointData *> points = {&pointData1, &pointData2, &pointData3, &pointData4, &pointData5, &pointData6};
    KDTree kdTree (points);
    kdTree.findNearestNeighbour({4, 4});

    //std::pair<int, int> neighbor = quadTree.getNearestNeighbour({4, 4});
    return 0;
}