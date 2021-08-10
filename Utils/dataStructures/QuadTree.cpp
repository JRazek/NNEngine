//
// Created by jrazek on 09.08.2021.
//

#include "QuadTree.h"
#include <algorithm>
#include <vector>

QuadTree::QuadTree(float posX, float posY, float sizeX, float sizeY, int pointsLimit, int levelLimit, int level,
                   QuadTree *parent)
        :
        posX(posX),
        posY(posY),
        sizeX(sizeX),
        sizeY(sizeY),
        pointCount(0),
        pointsLimit(pointsLimit),
        levelLimit(levelLimit),
        level(level),
        parent(parent) {

}

QuadTree::QuadTree(float sizeX, float sizeY, int levelLimit) : QuadTree(0, 0, sizeX, sizeY, 2, levelLimit, 0, nullptr) {

}

void QuadTree::insertPoint(float x, float y) {
    if(this->points.size()){

    }
}

QuadTree::~QuadTree() {
    if(!this->leaf){
        delete this->NW;
        delete this->NE;
        delete this->SW;
        delete this->SE;
    }
}

bool QuadTree::belongs(float x, float y) const {
    return this->posX <= x && this->posX + this->sizeX > x && this->posY <= y && this->posY + this->sizeY > y;
}


std::pair<int, int> QuadTree::getNearestNeighbour(const std::pair<int, int> &point) {
    //todo
}

bool QuadTree::belongs(const std::pair<int, int> &point) const {
    return belongs(point.first, point.second);
}

