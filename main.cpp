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


    //QuadTree quadTree(1024, 1024);
    int t = 3;

//    PointData pointData1({7,2}, &t);
//    PointData pointData2({5,4}, &t);
//    PointData pointData3({2,3}, &t);
//    PointData pointData4({9,6}, &t);
//    PointData pointData5({8,1}, &t);
//    PointData pointData6({4,7}, &t);
//    std::vector<PointData *> points = {&pointData1, &pointData2, &pointData3, &pointData4, &pointData5, &pointData6};


    std::vector<PointData *> points;
    for(int i = 0; i < 1e5; i ++){
        int randX = rand() % 10000;
        int randY = rand() % 10000;
        points.push_back(new PointData({randX, randY}));
    }
    KDTree kdTree (points);
    for(int i = 0; i < 1e4; i ++) {
        int randX = rand() % 10000;
        int randY = rand() % 10000;
        auto result = kdTree.findNearestNeighbour({randX, randY});
        std::cout << "x: " << result.first->point.first << " y: " << result.first->point.second<<"\n";
    }

    for(auto p : points){
        delete p;
    }
    //std::pair<int, int> neighbor = quadTree.getNearestNeighbour({4, 4});
    return 0;
}