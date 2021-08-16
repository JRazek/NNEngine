//
// Created by user on 16.08.2021.
//

#ifndef NEURALNETLIBRARY_VECTOR2F_H
#define NEURALNETLIBRARY_VECTOR2F_H
#include <vector>

struct Vector2f {
    float x, y;
    explicit Vector2f(const std::pair<float, float> &p);
    Vector2f(float _x, float _y);
};


#endif //NEURALNETLIBRARY_VECTOR2F_H
