//
// Created by jrazek on 10.08.2021.
//

#ifndef NEURALNETLIBRARY_POINTDATA_H
#define NEURALNETLIBRARY_POINTDATA_H
#include <vector>

template<typename T>
struct PointData {
    std::pair<float, float> point;
    T * data;
    /**
     *
     * @param point point where the data is located
     * @param data data itself
     * @note that no data is copied. This way its more efficient.
     * You should deallocate data by yourself if necessary.
     */
    PointData(const std::pair<float, float> &point, T * data);
};

template<typename T>
PointData<T>::PointData(const std::pair<float, float> &point, T *data):point(point), data(data) {

}


#endif //NEURALNETLIBRARY_POINTDATA_H
