//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_NETWORK_H
#define NEURALNETLIBRARY_NETWORK_H

#include "layers/Layer.h"



class Network {
    std::vector<Layer *> layers;
    /**
     * if first layer is convolution - use this feed method
     * @param data - data to convolve
     * @param h - height of tensor
     * @param w - width of tensor
     * @param d - depth of tensor
     */

    void feed(std::vector<byte> data, int h, int w, int d);

    /**
     * use this if and only if first layer is feed forward layer
     * @param data
     */
    void feed(std::vector<byte> data);
};


#endif //NEURALNETLIBRARY_NETWORK_H
