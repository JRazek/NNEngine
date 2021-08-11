//
// Created by jrazek on 11.08.2021.
//

#ifndef NEURALNETLIBRARY_KDTREE_H
#define NEURALNETLIBRARY_KDTREE_H
#include <vector>
#include "PointData.h"
struct KDTree {
    int median;
    bool dimension;

    KDTree * left;
    KDTree * right;

    bool leaf = false;
    /**
     * if leaf - this will have a val
     */
    PointData * point;
    /**
     *
     * @param pointsVec - dataset
     * @param dimension - 0 for x to split y for 1 to split
     */
    KDTree(std::vector<PointData *> pointsVec, bool dimension = 0);
    ~KDTree(){
        if(!leaf){
            delete left;
            delete right;
        }
    }
};


#endif //NEURALNETLIBRARY_KDTREE_H
