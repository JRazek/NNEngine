//
// Created by jrazek on 11.08.2021.
//

#ifndef NEURALNETLIBRARY_KDTREE_H
#define NEURALNETLIBRARY_KDTREE_H
#include <vector>
#include "PointData.h"
#include <cmath>
struct KDTree {
    /**
     * dimension in which the node is splitting data
     * false - x
     * true - y
     */
    bool dimension;

    /**
     * parent node
     */
    KDTree * const parent;


    /**
     * less or equal to left bigger to right
     */
    KDTree * leftChild = nullptr;
    KDTree * rightChild = nullptr;


    PointData * pointData;

    /**
     *
     * @param pointsVec - dataset
     * @param dimension - 0 for x to split y for 1 to split
     * @param segment - the segment on which the node is splitting data
     */
    explicit KDTree(std::vector<PointData *> pointsVec, bool dimension = false, KDTree * parent = nullptr);

    /**
     *
     * @param point finds nearest node from this pointData
     * @return pair of nearest node from specified pointData, distanceSquared squared from it.
     */
    std::pair<PointData *,  float> findNearestNeighbour(const std::pair<float, float> &point);

    ~KDTree();
};


#endif //NEURALNETLIBRARY_KDTREE_H
