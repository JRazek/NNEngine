//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "Network.h"
#include "layers/ConvolutionLayer.h"
#include "layers/FFLayer.h"
#include "../Utils/Bitmap.h"

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
    //this->layers.front()->run(&bitmap);
}

cn::Network::~Network() {
    for(auto l : this->layers){
        delete l;
    }
}

void cn::Network::appendConvolutionLayer(int kernelX, int kernelY, int kernelZ, int kernelsCount) {
    this->layers.push_back(new ConvolutionLayer(this->layers.size(), this, kernelsCount));
}

const std::vector<cn::Layer *> *cn::Network::getLayers() {
    return &this->layers;
}

cn::Network::Network(int w, int h, int d): dataWidth(w), dataHeight(h), dataDepth(d) {}
