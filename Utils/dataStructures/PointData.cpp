//
// Created by jrazek on 10.08.2021.
//
#include "PointData.h"

PointData::PointData(const std::pair<double, double> &_point, void * _data): point(_point), data(_data)
{}

PointData::PointData(const std::pair<double, double> &_point): point(_point){

}
