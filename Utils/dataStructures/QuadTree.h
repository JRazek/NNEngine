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
    const double posX, posY;
    const double sizeX, sizeY;
    const int level;

    QuadTree(double sizeX, double sizeY);

    void insertPoint(PointData &pointD);



    ~QuadTree();
    [[nodiscard]] bool belongs(double x, double y) const;
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

    QuadTree(double posX, double posY, double sizeX, double sizeY, int level,
             QuadTree *parent);

};


#endif //NEURALNETLIBRARY_QUADTREE_H
