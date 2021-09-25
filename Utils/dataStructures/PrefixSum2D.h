//
// Created by user on 29.08.2021.
//

#ifndef NEURALNETLIBRARY_PREFIXSUM2D_H
#define NEURALNETLIBRARY_PREFIXSUM2D_H
#include "Tensor.h"
namespace cn {
    template<typename T>
    class PrefixSum2D : public cn::Tensor<T> {
    public:
        template<typename Y>
        PrefixSum2D(const cn::Tensor<Y> &input);

        T sum(Vector3<int> nw, Vector3<int> se);
    };
}

template<typename T>
template<typename Y>
cn::PrefixSum2D<T>::PrefixSum2D(const cn::Tensor<Y> &input):cn::Tensor<T>(input.w(), input.h(), input.d()) {
    for(int c = 0; c < this->d(); c++) {
        for (int x = 0; x < this->w(); x++) {
            this->setCell(x, 0, c, input.getCell(x, 0, c));
        }
        for (int y = 0; y < this->h(); y++) {
            this->setCell(0, y, c, input.getCell(0, y, c));
        }
    }
    for(int z = 0; z < input.d(); z++){
        for(int y = 1; y < input.h(); y++){
            for(int x = 1; x < input.w(); x++){
                this->setCell(x, y, z, this->getCell(x - 1, y - 1, z) + sum({0, y, z}, {x - 1, y, z}) + sum({x, 0, z}, {x, y - 1, z}));
            }
        }
    }
}
template<typename T>
T cn::PrefixSum2D<T>::sum(Vector3<int> nw, Vector3<int> se) {
    if(nw.x == 0 && nw.y == 0){
        return this->getCell(se);
    }else if(nw.x == 0){
        return this->getCell(se) - this->getCell(se.x, nw.y - 1, nw.z);
    }else if(nw.y == 0){
        return this->getCell(se) - this->getCell(nw.x - 1, se.y, nw.z);
    }
    return this->getCell(se) - this->getCell(se.x - 1, se.y, se.z) - this->getCell(se.x, se.y - 1, se.z) + this->getCell(nw - Vector3<int>(1, 1, 0));
}


#endif //NEURALNETLIBRARY_PREFIXSUM2D_H
