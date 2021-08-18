//
// Created by jrazek on 27.07.2021.
//

#include "FFLayer.h"
#include "../Network.h"
cn::FFLayer::FFLayer(int _id, Network *_network, int _inputSize) : cn::Layer(_id, _network) {

}

void cn::FFLayer::run(const Bitmap<float> &bitmap) {

}

void cn::FFLayer::randomInit() {
    for(auto &w : weights){
        w = network->getWeightRandom();
    }
}
