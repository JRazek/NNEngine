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
     * @param _pointsVec - dataset
     * @param _dimension - 0 for x to split y for 1 to split
     * @param segment - the segment on which the node is splitting data
     */
    explicit KDTree(std::vector<PointData *> _pointsVec, bool _dimension = false, KDTree * _parent = nullptr);

    /**
     *
     * @param point finds nearest node from this pointData
     * @return pair of nearest node from specified pointData, distanceSquared squared from it.
     */
    std::pair<PointData *,  double> findNearestNeighbour(const std::pair<double, double> &point);

    ~KDTree();
};


#endif //NEURALNETLIBRARY_KDTREE_H
