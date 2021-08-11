//
// Created by jrazek on 11.08.2021.
//

#ifndef NEURALNETLIBRARY_KDTREE_H
#define NEURALNETLIBRARY_KDTREE_H
#include <vector>
#include "PointData.h"
struct KDTree {
    bool dimension;


    KDTree * leftChild = nullptr;
    KDTree * rightChild = nullptr;


    PointData * point;
    /**
     *
     * @param pointsVec - dataset
     * @param dimension - 0 for x to split y for 1 to split
     */
    KDTree(std::vector<PointData *> pointsVec, bool dimension = 0);

    /**
     *
     * @param point finds nearest node from this point
     * @return pair of nearest node from specified point, distance from it.
     */
    std::pair<PointData *,  float> findNearestNeighbour(const std::pair<float, float> &point);

    ~KDTree();
};


#endif //NEURALNETLIBRARY_KDTREE_H
