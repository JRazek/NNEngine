//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include <iostream>
#include "Network.h"
#include "layers/ConvolutionLayer.h"
#include "layers/FFLayer.h"

void cn::Network::appendLayer(cn::FFLayer * layer) {
    this->layers.push_back(layer);
}

void cn::Network::appendLayer(ConvolutionLayer * layer) {
    if(!this->layers.empty()){
        if(auto * l = dynamic_cast<cn::FFLayer *>(this->layers.back())) {
            throw std::invalid_argument("cannot use convolution layer after ff layer!");
        }else{
            this->layers.push_back(layer);
        }
    }
}

void cn::Network::feed(const byte *input) {
    this->data = input;
    cn::Bitmap<byte> bitmap(this->dataWidth, this->dataHeight, this->dataHeight, input);
    //normalize image
    cn::Bitmap<float> * normalized = cn::Utils::normalize(bitmap);

    this->layers.front()->run(normalized);

    delete normalized;
}

cn::Network::~Network() {
    for(auto l : this->layers){
        delete l;
    }
}

void cn::Network::appendConvolutionLayer(int kernelX, int kernelY, int kernelZ, int kernelsCount) {
    this->layers.push_back(new ConvolutionLayer(this->layers.size(), this, kernelsCount, 0, 0, 0, 0, 0));
}

const std::vector<cn::Layer *> *cn::Network::getLayers() {
    return &this->layers;
}

cn::Network::Network(int w, int h, int d): dataWidth(w), dataHeight(h), dataDepth(d) {
    this->data = new byte [w * h * d];
}
