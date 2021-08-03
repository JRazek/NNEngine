//
// Created by jrazek on 27.07.2021.
//

#include "ConvolutionLayer.h"
#include "../Network.h"
#include "../Bitmap.h"

void ConvolutionLayer::run(Bitmap *bitmap) {

}



ConvolutionLayer::ConvolutionLayer(int id, Network *network, int w, int h, int d, int kernelsCount) : Layer(id, network) {

}

Bitmap *ConvolutionLayer::convolve(const Bitmap *kernel, const Bitmap *input, int padding) {
    //convolution here
    return nullptr;
}
