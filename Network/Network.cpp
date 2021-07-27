//
// Created by jrazek on 27.07.2021.
//

#include "Network.h"
#include "layers/ConvolutionLayer.h"

void Network::appendLayer(FFLayer *layer) {

}

void Network::appendLayer(ConvolutionLayer * layer) {
    if(!this->layers.empty()){
        if(auto * l = dynamic_cast<ConvolutionLayer *>(this->layers.back())) {
            this->layers.push_back(layer);
        }
    }
}
