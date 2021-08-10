//
// Created by jrazek on 09.08.2021.
//

#include "QuadTree.h"
#include <algorithm>
#include <vector>

QuadTree::QuadTree(float posX, float posY, float sizeX, float sizeY, int pointsLimit, int levelLimit, int level,
                   QuadTree *parent)
        : posX(posX),
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
    if(belongs(x, y)){
        if(pointCount > pointsLimit && this->isLeaf && level < levelLimit - 1){
            this->isLeaf = false;
            this->NW = new QuadTree(this->posX, this->posY, this->sizeX / 2, this->sizeY / 2, pointsLimit, levelLimit, level + 1, this);
            this->NE = new QuadTree(this->posX + this->sizeX / 2, this->posY, this->sizeX / 2, this->sizeY / 2,
                                    pointsLimit, levelLimit, level + 1, this);
            this->SW = new QuadTree(this->posX, this->posY + this->sizeY / 2, this->sizeX / 2, this->sizeY / 2,
                                    pointsLimit, levelLimit, level + 1, this);
            this->SE = new QuadTree(this->posX + this->sizeX / 2, this->posY + this->sizeY / 2, this->sizeX / 2,
                                    this->sizeY / 2, pointsLimit, levelLimit, level + 1, this);
            for(auto &pX : points){
                for(auto &pY : pX.second){
                    this->insertPoint(pX.first, pY);
                }
            }
            this->points.clear();
        }
        if(!this->isLeaf){
            this->NW->insertPoint(x, y);
            this->NE->insertPoint(x, y);
            this->SW->insertPoint(x, y);
            this->SE->insertPoint(x, y);

            this->pointCount = this->NW->pointCount + this->NE->pointCount + this->SW->pointCount + this->SE->pointCount;
        }else{
            points[x].insert(y);
            pointCount ++;
        }
    }
}

QuadTree::~QuadTree() {
    if(!this->isLeaf){
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
        if(!this->isLeaf){
            this->NW->removePoint(x, y);
            this->NE->removePoint(x, y);
            this->SW->removePoint(x, y);
            this->SE->removePoint(x, y);
            if(!this->pointCount){
                delete this->NW;
                delete this->NE;
                delete this->SW;
                delete this->SE;
                this->isLeaf = true;
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
    if(!this->isLeaf) {
        this->NW->getChildrenPoints(pointsSet);
        this->NE->getChildrenPoints(pointsSet);
        this->SW->getChildrenPoints(pointsSet);
        this->SE->getChildrenPoints(pointsSet);
    }else{
        pointsSet.insert(this->points.begin(), this->points.end());
    }
}

std::pair<int, int> QuadTree::getNearestNeighbour(const std::pair<int, int> &point) {
    std::vector<std::pair<int, int>> candidates;
    if(this->belongs(point)){
        QuadTree * leafContaining = getLeafContainingPoint(point);
        for(auto &pX : leafContaining->points){
            for(auto pY : pX.second){
                candidates.push_back({pX.first, pY});
            }
        }
        QuadTree * parentNode = leafContaining->parent;
        while(parentNode != nullptr){
            
            parentNode = parentNode->parent;
        }
    }
    return std::pair<int, int>();
}

QuadTree *QuadTree::getLeafContainingPoint(const std::pair<int, int> &point) {
    if(this->belongs(point)){
        if(this->isLeaf)
            return this;
        else{
            QuadTree * result;
            result = this->NW->getLeafContainingPoint(point);
            if(result != nullptr)
                return result;
            result = this->NE->getLeafContainingPoint(point);
            if(result != nullptr)
                return result;
            result = this->SW->getLeafContainingPoint(point);
            if(result != nullptr)
                return result;
            result = this->SE->getLeafContainingPoint(point);
            if(result != nullptr)
                return result;
        }
        throw std::logic_error("ERROR!!!");
    }
    return nullptr;
}

bool QuadTree::belongs(const std::pair<int, int> &point) const {
    return belongs(point.first, point.second);
}

