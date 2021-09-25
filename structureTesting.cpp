//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/dataStructures/Tensor.h"
#include "Utils/Utils.h"
#include "Utils/dataStructures/QuadTree.h"
#include "Utils/dataStructures/KDTree.h"
#include "Network/Network.h"
#include "Network/layers/ConvolutionLayer/ConvolutionLayer.h"
#include <opencv2/opencv.hpp>
#include <fstream>

int main(){
    cv::Mat mat = cv::imread("resources/aPhoto.jpg");
    cn::Network network(800, 800, 3, 0);


    //QuadTree quadTree(1024, 1024);

//    PointData pointData1({7,2}, &t);
//    PointData pointData2({5,4}, &t);
//    PointData pointData3({2,3}, &t);
//    PointData pointData4({9,6}, &t);
//    PointData pointData5({8,1}, &t);
//    PointData pointData6({4,7}, &t);
//    std::vector<PointData *> points = {&pointData1, &pointData2, &pointData3, &pointData4, &pointData5, &pointData6};

    int pointsCount, testsCount;
    std::cin >> pointsCount >> testsCount;

    std::vector<PointData *> points;
    points.reserve(pointsCount);

    std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> tests;//test, answer
    tests.reserve(pointsCount);
    for(int i = 0; i < pointsCount; i ++){
        double x,y;
        std::cin >> x >> y;
        points.push_back(new PointData({x, y}));
    }
    for(int i = 0; i < testsCount; i ++){
        double xt,yt,xa,ya;
        std::cin >> xt >> yt >> xa >> ya;
        tests.push_back({{xt, yt}, {xa, ya}});
    }

    KDTree kdTree(points);
    for(int i = 0; i < testsCount; i ++){
        auto p = tests[i];
        auto res = kdTree.findNearestNeighbour(p.first);
        double distSquaredTest = pow(p.first.first - p.second.first, 2) +  pow(p.first.second - p.second.second, 2);
        if(res.second == distSquaredTest){
            //std::cout<<"OK \n";
        }else if (res.second < distSquaredTest){
            //std::cout<<"BETTER? \n";
        }else {
            std::cout<<"WRONG \n";
        }

    }
    for(auto p : points){
        delete p;
    }
    //std::pair<int, int> neighbor = quadTree.getNearestNeighbour({4, 4});
    return 0;
}