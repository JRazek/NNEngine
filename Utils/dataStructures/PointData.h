//
// Created by jrazek on 10.08.2021.
//

#ifndef NEURALNETLIBRARY_POINTDATA_H
#define NEURALNETLIBRARY_POINTDATA_H
#include <vector>

struct PointData {
    std::pair<float, float> point;
    void * data;
    /**
     *
     * @param point point where the data is located
     * @param data data itself
     * @note that no data is copied. This way its more efficient.
     * You should deallocate data by yourself if necessary.
     */
    PointData(const std::pair<float, float> &point, void * data);
    PointData(const std::pair<float, float> &point);
};



#endif //NEURALNETLIBRARY_POINTDATA_H
