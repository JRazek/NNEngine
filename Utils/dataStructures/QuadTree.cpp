//
// Created by jrazek on 09.08.2021.
//

#include "QuadTree.h"
#include <algorithm>
#include <vector>

void QuadTree::insertPoint(PointData &pointD) {
    if(this->belongs(pointD.point)){
        if(this->type == 0){
            this->pointData = &pointD;
            this->type = 1;
        }else{
            if(this->type == 1) {
                if(this->pointData->point == pointD.point){
                    throw std::logic_error("SAME POINTS!");
                }
                this->NW = new QuadTree(posX, posY,  sizeX / 2,  sizeY / 2, level + 1, this);
                this->NE = new QuadTree(posX + sizeX / 2, posY,  sizeX / 2,  sizeY / 2, level + 1, this);
                this->SW = new QuadTree(posX, posY + sizeY / 2,  sizeX / 2,  sizeY / 2, level + 1, this);
                this->SE = new QuadTree(posX + sizeX / 2, posY + sizeY / 2,  sizeX / 2,  sizeY / 2, level + 1, this);

                this->type = 2;
                this->NW->insertPoint(*this->pointData);
                this->NE->insertPoint(*this->pointData);
                this->SW->insertPoint(*this->pointData);
                this->SE->insertPoint(*this->pointData);
                this->pointData = nullptr;
            }
            this->NW->insertPoint(pointD);
            this->NE->insertPoint(pointD);
            this->SW->insertPoint(pointD);
            this->SE->insertPoint(pointD);
        }
    }
}

QuadTree::~QuadTree() {
    if(this->type == 2){
        delete this->NW;
        delete this->NE;
        delete this->SW;
        delete this->SE;
    }
}

bool QuadTree::belongs(double x, double y) const {
    return this->posX <= x && this->posX + this->sizeX > x && this->posY <= y && this->posY + this->sizeY > y;
}
bool QuadTree::belongs(const std::pair<int, int> &point) const {
    return belongs(point.first, point.second);
}

PointData * QuadTree::getNearestNeighbour(const std::pair<int, int> &point) {
    if(point.first)
        return nullptr;
    else
        return NULL;
}

QuadTree::QuadTree(double posX, double posY, double sizeX, double sizeY, int level, QuadTree *parent):
        posX(posX),
        posY(posY),
        sizeX(sizeX),
        sizeY(sizeY),
        level(level),
        parent(parent),
        type(0)
        {}

QuadTree::QuadTree(double sizeX, double sizeY):
        QuadTree(0, 0, sizeX, sizeY, 0, nullptr)
        {}

