//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
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
