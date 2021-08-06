//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include <iostream>
#include "Network.h"
#include "layers/ConvolutionLayer.h"
#include "layers/FFLayer.h"
#include "../Utils/Utils.h"

void cn::Network::appendLayer(cn::Layer * layer) {
    //todo validation!
    this->layers.push_back(layer);
}

void cn::Network::feed(const byte *input) {
    cn::Bitmap<byte> bitmap(this->dataWidth, this->dataHeight, this->dataHeight, input, 0);
    //normalize image
    cn::Bitmap<float> normalized = cn::Utils::normalize(bitmap);

    this->layers.front()->run(normalized);
}

cn::Network::~Network() {
    for(auto l : this->layers){
        delete l;
    }
}

void cn::Network::appendConvolutionLayer(int kernelX, int kernelY, int kernelZ, int kernelsCount, int paddingX,
                                         int paddingY) {
    this->layers.push_back(new ConvolutionLayer(this->layers.size(), this, kernelX, kernelY, kernelZ, kernelsCount, paddingX, paddingY));
}

const std::vector<cn::Layer *> *cn::Network::getLayers() {
    return &this->layers;
}

cn::Network::Network(int w, int h, int d): dataWidth(w), dataHeight(h), dataDepth(d) {
}
