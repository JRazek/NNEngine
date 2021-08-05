//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_NETWORK_H
#define NEURALNETLIBRARY_NETWORK_H

#include <vector>
#include "Bitmap.h"

class Layer;
class FFLayer;
class ConvolutionLayer;

typedef unsigned char byte;

class Network {
protected:
    std::vector<Layer *> layers;
    friend class Layer;


    const byte * data;
    /**
     * what the dimensions of the byte array is after being normalized
     */
    const int dataWidth;
    const int dataHeight;
    const int dataDepth;
    friend ConvolutionLayer;
public:

    void appendConvolutionLayer(int kernelX, int kernelY, int kernelZ, int kernelsCount);

    void appendLayer(FFLayer  * layer);
    void appendLayer(ConvolutionLayer * layer);

    /**
     * if first layer is convolution - use this feed method
     * @param data - dataP to convolve
     * @param w - width of tensor
     * @param h - height of tensor
     * @param d - depth of tensor
     */

    void feed(const byte *input);

    /**
     * use this if and only if first layer is convolution
     * @param data
     */

    void feed(const Bitmap<float> * const bitmap);


    const std::vector<Layer *> * getLayers();

    /**
     * input size for feed
     * @param w width
     * @param h height
     * @param d depth
     *
     * if the first layer is image - set all the properties.
     * In case of using only FFLayers - set height and depth to 1.
     */

    Network(int w, int h, int d);

    ~Network();
};


#endif //NEURALNETLIBRARY_NETWORK_H
