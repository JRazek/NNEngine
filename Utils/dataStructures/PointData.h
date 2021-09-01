//
// Created by jrazek on 10.08.2021.
//

#ifndef NEURALNETLIBRARY_POINTDATA_H
#define NEURALNETLIBRARY_POINTDATA_H
#include <vector>

struct PointData {
    std::pair<double, double> point;
    void * data;
    /**
     *
     * @param _point pointData where the data is located
     * @param _data data itself
     * @note that no data is copied. This way its more efficient.
     * You should deallocate data by yourself if necessary.
     */
    PointData(const std::pair<double, double> &_point, void * _data);
    PointData(const std::pair<double, double> &_point);
};



#endif //NEURALNETLIBRARY_POINTDATA_H
