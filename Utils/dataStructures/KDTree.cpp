//
// Created by jrazek on 11.08.2021.
//

#include "KDTree.h"
#include <algorithm>

KDTree::KDTree(std::vector<PointData *> pointsVec, bool dimension) {
    if(pointsVec.size() == 1){
        this->pointData = pointsVec.front();
        return;
    }
    this->dimension = dimension;
    if(dimension == 0) {
        std::sort(pointsVec.begin(), pointsVec.end(), [](const PointData *p1, const PointData *p2) {
            return p1->point.first < p2->point.first;
        });
    }else{
        std::sort(pointsVec.begin(), pointsVec.end(), [](const PointData *p1, const PointData *p2) {
            return p1->point.second < p2->point.second;
        });
    }
    this->pointData = pointsVec[pointsVec.size() / 2];
    std::vector<PointData *> leftVec;
    std::vector<PointData *> rightVec;
    leftVec.reserve(pointsVec.size()/2);
    rightVec.reserve(pointsVec.size()/2);

    bool afterMid = false;
    for(auto p : pointsVec){
        if(p != this->pointData){
            if(!afterMid)
                leftVec.push_back(p);
            else
                rightVec.push_back(p);
        }
        else {
            afterMid = true;
        }
    }
    if(leftVec.size())
        this->leftChild = new KDTree(leftVec, !dimension);
    if(rightVec.size())
        this->rightChild = new KDTree(rightVec, !dimension);
}

KDTree::~KDTree() {
    if(leftChild != nullptr)
        delete leftChild;
    if(rightChild != nullptr)
        delete rightChild;
}

std::pair<PointData *,  float> KDTree::findNearestNeighbour(const std::pair<float, float> &pointSearch) {
    return {nullptr, 2};
}
