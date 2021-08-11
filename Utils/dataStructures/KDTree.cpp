//
// Created by jrazek on 11.08.2021.
//

#include "KDTree.h"
#include <algorithm>
#include <cmath>

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
    auto it = pointsVec.begin() + pointsVec.size() / 2;
    while(it+1 != pointsVec.end()){
        if(!dimension) {
            if ((*it)->point.first == (*it+1)->point.first) {
                ++it;
            } else
                break;
        }else{
            if ((*it)->point.second == (*it+1)->point.second) {
                ++it;
            } else
                break;
        }
    }
    this->pointData = (*it);
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
    if(!leftVec.empty())
        this->leftChild = new KDTree(leftVec, !dimension);
    if(!rightVec.empty())
        this->rightChild = new KDTree(rightVec, !dimension);
}

KDTree::~KDTree() {
    delete leftChild;
    delete rightChild;
}

std::pair<PointData *, float> KDTree::findNearestNeighbour(const std::pair<float, float> &pointSearch) {
    std::pair<PointData *, float> nearest;
    std::pair<PointData *, float> anotherNearest;
    if(!dimension){
        //x dim
        if(pointSearch.first <= this->pointData->point.first){
            //go to left
            if(this->leftChild == nullptr)
                return {this->pointData, std::pow(pointSearch.first - this->pointData->point.first, 2)};
            nearest = this->leftChild->findNearestNeighbour(pointSearch);

            if(this->rightChild != nullptr && nearest.second >= pow(this->pointData->point.first - pointSearch.first, 2)){
                anotherNearest = this->rightChild->findNearestNeighbour(pointSearch);
            }

        }else{
            if(this->rightChild == nullptr)
                return {this->pointData, std::pow(pointSearch.first - this->pointData->point.first, 2)};
            nearest = this->rightChild->findNearestNeighbour(pointSearch);

            if(this->leftChild != nullptr && nearest.second >= pow(this->pointData->point.first - pointSearch.first, 2)){
                anotherNearest = this->leftChild->findNearestNeighbour(pointSearch);
            }
        }
    }else{
        if(pointSearch.second <= this->pointData->point.second){
            //go to left
            if(this->leftChild == nullptr)
                return {this->pointData, std::pow(pointSearch.second - this->pointData->point.second, 2)};
            nearest = this->leftChild->findNearestNeighbour(pointSearch);

            if(this->rightChild != nullptr && nearest.second >= pow(this->pointData->point.second - pointSearch.second, 2)){
                anotherNearest = this->rightChild->findNearestNeighbour(pointSearch);
            }
        }else{
            if(this->rightChild == nullptr)
                return {this->pointData, std::pow(pointSearch.second - this->pointData->point.second, 2)};
            nearest = this->rightChild->findNearestNeighbour(pointSearch);

            if(this->leftChild != nullptr && nearest.second >= pow(this->pointData->point.second - pointSearch.second, 2)){
                anotherNearest = this->leftChild->findNearestNeighbour(pointSearch);
            }
        }
    }

    //here you have alleged nearest point

    if(anotherNearest.second < nearest.second)
        return anotherNearest;
    return nearest;
}
