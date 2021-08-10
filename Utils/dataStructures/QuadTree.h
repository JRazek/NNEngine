//
// Created by jrazek on 09.08.2021.
//

#ifndef NEURALNETLIBRARY_QUADTREE_H
#define NEURALNETLIBRARY_QUADTREE_H
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "PointData.h"

class QuadTree {
public:
    const float posX, posY;
    const float sizeX, sizeY;
    const int level;

    QuadTree(float sizeX, float sizeY);

    void insertPoint(PointData &pointD);



    ~QuadTree();
    [[nodiscard]] bool belongs(float x, float y) const;
    [[nodiscard]] bool belongs(const std::pair<int, int> &point) const;

    PointData * getNearestNeighbour(const std::pair<int, int> &point);
private:

    PointData * pointData;

    QuadTree * parent = nullptr;

    /**
     * 0 - empty leaf
     * 1 - leaf with data
     * 2 - parent
     */
    int type;
    QuadTree * NW;
    QuadTree * NE;
    QuadTree * SW;
    QuadTree * SE;

    QuadTree(float posX, float posY, float sizeX, float sizeY, int level,
             QuadTree *parent);

};


#endif //NEURALNETLIBRARY_QUADTREE_H
