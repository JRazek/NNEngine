//
// Created by jrazek on 27.07.2021.
//

#include "FFLayer.h"
#include "../Network.h"

cn::FFLayer::FFLayer(int _id, const DifferentiableFunction &_differentiableFunction, Network *_network, int _neuronsCount):
        cn::Layer(_id, _network),
        neuronsCount(_neuronsCount),
        differentiableFunction(_differentiableFunction),
        weights(_neuronsCount),
        biases(_neuronsCount),
        outputs(_neuronsCount)
{}

void cn::FFLayer::run(const Bitmap<float> &bitmap) {

}

void cn::FFLayer::randomInit() {
    for(auto &w : weights){
        w = network->getWeightRandom();
    }
    for(auto &b : biases){
        b = network->getWeightRandom();
    }
}
