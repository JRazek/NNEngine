//
// Created by jrazek on 11.08.2021.
//

#include "KDTree.h"
#include <algorithm>
#include <cmath>
#include "../Utils.h"

KDTree::KDTree(std::vector<PointData *> pointsVec, bool dimension, KDTree * const parent) : parent(parent) {
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
        this->leftChild = new KDTree(leftVec, !dimension, this);
    if(!rightVec.empty())
        this->rightChild = new KDTree(rightVec, !dimension, this);
}

KDTree::~KDTree() {
    delete leftChild;
    delete rightChild;
}

std::pair<PointData *, float> KDTree::findNearestNeighbour(const std::pair<float, float> &pointSearch) {
    std::vector <std::pair<PointData *, float>> nearest(3);
    nearest[0] = {this->pointData, cn::Utils::distanceSquared(pointSearch, this->pointData->point)};
    nearest[1] = {nullptr, INFINITY};
    nearest[2] = {nullptr, INFINITY};

    bool side; // 0 - left, 1 - right

    if((!dimension && pointSearch.first <= pointData->point.first) || (dimension && pointSearch.second <= pointData->point.second)) {
        //go to left
        side = 0;
        if(this->leftChild != nullptr) {
            nearest[1] = this->leftChild->findNearestNeighbour(pointSearch);
        }
        else {
            nearest[1] = {pointData, cn::Utils::distanceSquared(pointSearch, pointData->point)};
        }
    }else{
        //go to right
        side = 1;
        if(this->rightChild != nullptr) {
            nearest[1] = this->rightChild->findNearestNeighbour(pointSearch);
        }
        else {
            nearest[1] = {pointData, cn::Utils::distanceSquared(pointSearch, pointData->point)};
        }

    }

    std::sort(nearest.begin(), nearest.end(), [](auto p1, auto p2){
        return p1.second < p2.second;
    });
    //find better on the other side

    KDTree * another = side == 0 ? this->rightChild : this->leftChild;
    if(another != nullptr) {
        if (!dimension) {
            //todo fix that check
            if (nearest[0].second > pow(this->pointData->point.first - pointSearch.first, 2)){
                nearest[2] = another->findNearestNeighbour(pointSearch);//doesnt matter if not first after sort
            }
        } else {
            //todo fix that check
            if (nearest[0].second > pow(this->pointData->point.second - pointSearch.second, 2)){
                nearest[2] = another->findNearestNeighbour(pointSearch);
            }
        }
    }
    std::sort(nearest.begin(), nearest.end(), [](auto p1, auto p2){
        return p1.second < p2.second;
    });

    return nearest[0];
}
