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


    PointData * pointData;
    /**
     *
     * @param pointsVec - dataset
     * @param dimension - 0 for x to split y for 1 to split
     */
    explicit KDTree(std::vector<PointData *> pointsVec, bool dimension = false);

    /**
     *
     * @param point finds nearest node from this pointData
     * @return pair of nearest node from specified pointData, distance from it.
     */
    std::pair<PointData *,  float> findNearestNeighbour(const std::pair<float, float> &point);

    ~KDTree();
};


#endif //NEURALNETLIBRARY_KDTREE_H
