//
// Created by jrazek on 11.08.2021.
//

#include "KDTree.h"
#include <algorithm>
#include <iostream>

KDTree::KDTree(std::vector<PointData *> pointsVec, bool dimension) {
    if(pointsVec.size() == 1){
        this->leaf = 1;
        this->point = pointsVec.front();
        return;
    }
    this->dimension = dimension;
    if(dimension == 0) {
        std::sort(pointsVec.begin(), pointsVec.end(), [](const PointData *p1, const PointData *p2) {
            return p1->point.first < p2->point.first;
        });
        this->median = pointsVec[pointsVec.size() / 2]->point.first;
    }else{
        std::sort(pointsVec.begin(), pointsVec.end(), [](const PointData *p1, const PointData *p2) {
            return p1->point.second < p2->point.second;
        });
        this->median = pointsVec[pointsVec.size() / 2]->point.second;
    }
    PointData tmpPoint()
    std::vector<PointData *> leftVec;
    std::vector<PointData *> rightVec;
    auto mid = std::lower_bound(pointsVec.begin(), pointsVec.end(), )
}
