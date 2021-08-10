//
// Created by jrazek on 09.08.2021.
//

#include "QuadTree.h"
#include <algorithm>
#include <vector>




void QuadTree::insertPoint(float x, float y) {

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
bool QuadTree::belongs(const std::pair<int, int> &point) const {
    return belongs(point.first, point.second);
}

PointData<void> QuadTree::getNearestNeighbour(const std::pair<int, int> &point) {

}

QuadTree::QuadTree(float posX, float posY, float sizeX, float sizeY, int level, QuadTree *parent):
        posX(posX),
        posY(posY),
        sizeX(sizeX),
        sizeY(sizeY),
        level(level),
        parent(parent)
        {}

QuadTree::QuadTree(float sizeX, float sizeY):
        QuadTree(0, 0, sizeX, sizeY, 0, nullptr)
        {}

