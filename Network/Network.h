//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_NETWORK_H
#define NEURALNETLIBRARY_NETWORK_H

#include <vector>

class Layer;
class FFLayer;
class ConvolutionLayer;

typedef unsigned char byte;

class Network {
protected:
    std::vector<Layer *> layers;
    friend class Layer;
public:

    void appendLayer(FFLayer  * layer);
    void appendLayer(ConvolutionLayer * layer);

    /**
     * if first layer is convolution - use this feed method
     * @param data - data to convolve
     * @param w - width of tensor
     * @param h - height of tensor
     * @param d - depth of tensor
     */

    void feed(std::vector<byte> data, int w, int h, int d);

    /**
     * use this if and only if first layer is feed forward layer
     * @param data
     */
    void feed(std::vector<byte> data);

    const std::vector<Layer *> * getLayers();
};


#endif //NEURALNETLIBRARY_NETWORK_H
