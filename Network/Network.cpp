//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include <iostream>
#include "Network.h"
#include "layers/ConvolutionLayer.h"
#include "layers/FFLayer.h"

void Network::appendLayer(FFLayer * layer) {
    this->layers.push_back(layer);
}

void Network::appendLayer(ConvolutionLayer * layer) {
    if(!this->layers.empty()){
        if(auto * l = dynamic_cast<FFLayer *>(this->layers.back())) {
            throw std::invalid_argument("cannot use convolution layer after ff layer!");
        }else{
            this->layers.push_back(layer);
        }
    }
}

void Network::feed(const byte *input) {
    this->data = input;
    Bitmap bitmap(this->dataWidth, this->dataHeight, this->dataHeight, input);
    this->layers.front()->run(&bitmap);
}

Network::~Network() {
    for(auto l : this->layers){
        delete l;
    }
}

void Network::appendConvolutionLayer(int kernelX, int kernelY, int kernelZ, int kernelsCount) {
    this->layers.push_back(new ConvolutionLayer(this->layers.size(), this, kernelsCount));
}

const std::vector<Layer *> *Network::getLayers() {
    return &this->layers;
}

Network::Network(int w, int h, int d):dataWidth(w), dataHeight(h), dataDepth(d) {
    this->data = new byte [w * h * d];
}
