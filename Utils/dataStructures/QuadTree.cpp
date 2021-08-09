//
// Created by jrazek on 09.08.2021.
//

#include "QuadTree.h"

QuadTree::QuadTree(float posX, float posY, float sizeX, float sizeY, int pointsLimit, QuadTree * parent)
        : posX(posX), posY(posY), sizeX(sizeX), sizeY(sizeY), pointCount(0), pointsLimit(pointsLimit), parent(parent) {

}

QuadTree::QuadTree(float sizeX, float sizeY): QuadTree(0, 0, sizeX, sizeY, 0, nullptr) {

}

void QuadTree::insertPoint(float x, float y) {
    if(belongs(x, y)){
        pointCount ++;
        if(pointCount > pointsLimit && this->NW == nullptr){
            this->NW = new QuadTree(this->posX, this->posY, this->sizeX / 2, this->sizeY / 2, pointsLimit, this);
            this->NE = new QuadTree(this->posX + this->sizeX / 2, this->posY, this->sizeX / 2, this->sizeY / 2, pointsLimit, this);
            this->SW = new QuadTree(this->posX, this->posY + this->sizeY / 2, this->sizeX / 2, this->sizeY / 2, pointsLimit, this);
            this->SE = new QuadTree(this->posX + this->sizeX / 2, this->posY + this->sizeY / 2, this->sizeX / 2,
                                    this->sizeY / 2, pointsLimit, this);
            for(auto &pX : points){
                for(auto &pY : pX.second){
                    this->insertPoint(pX.first, pY);
                }
            }
            this->points.clear();
        }
        if(this->NW != nullptr){
            this->NW->insertPoint(x, y);
            this->NE->insertPoint(x, y);
            this->SW->insertPoint(x, y);
            this->SE->insertPoint(x, y);
        }else{
            points[x].insert(y);
        }
    }
}

QuadTree::~QuadTree() {
    if(this->NW != nullptr){
        delete this->NW;
        delete this->NE;
        delete this->SW;
        delete this->SE;
    }
}

bool QuadTree::belongs(float x, float y) const {
    return this->posX <= x && this->posX + this->sizeX > x && this->posY <= y && this->posY + this->sizeY > y;
}

void QuadTree::removePoint(float x, float y) {
    if(belongs(x, y)){
        this->pointCount --;
        if(this->NW != nullptr){
            this->NW->removePoint(x, y);
            this->NE->removePoint(x, y);
            this->SW->removePoint(x, y);
            this->SE->removePoint(x, y);
            if(!this->pointCount){
                delete this->NW;
                delete this->NE;
                delete this->SW;
                delete this->SE;
                this->NW = nullptr;
                this->NE = nullptr;
                this->SW = nullptr;
                this->SE = nullptr;
            }
        }else{
            if(points.find(x) != points.end())
                points[x].erase(y);
            if(points[x].empty()){
                points.erase(x);
            }
        }
    }
}


void QuadTree::getChildrenPoints(std::unordered_map<float, std::unordered_set<float>> &pointsSet) {
    if(this->NW != nullptr) {
        this->NW->getChildrenPoints(pointsSet);
        this->NE->getChildrenPoints(pointsSet);
        this->SW->getChildrenPoints(pointsSet);
        this->SE->getChildrenPoints(pointsSet);
    }else{
        pointsSet.insert(this->points.begin(), this->points.end());
    }
}

