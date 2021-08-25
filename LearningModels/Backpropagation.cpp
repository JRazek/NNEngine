//
// Created by user on 21.08.2021.
//

#include "Backpropagation.h"
#include "../Utils/Bitmap.h"
#include "../Network/Network.h"
cn::Backpropagation::Backpropagation(cn::Network &_network): network(_network) {

}

void cn::Backpropagation::propagate(const cn::Bitmap<float> &target) {
    OutputLayer *layer = network.getOutputLayer();
    layer->setTarget(&target);
    const Bitmap<float> &output = *layer->getOutput();
    if(output.w() != target.w() || output.h() != target.h() || output.d() != target.d()){
        throw std::logic_error("Backpropagation, invalid target!");
    }
    float error = 0;
    for(int i = 0; i < target.w(); i ++){
        error += std::pow(target.getCell(i, 0, 0) - output.getCell(i, 0, 0), 2);
    }
    for(Learnable *learnable : *network.getLearnables()){
        for(int n = 0; n < learnable->getNeuronsCount();  n++){
            for(int w = 0; w < learnable->weightsCount() / learnable->getNeuronsCount(); w ++){
                float oldWeight = learnable->getWeight(n, w);
                float delta = -0.5 * learnable->diffWeight(n, w);
                learnable->setWeight(n, w, oldWeight + delta);
            }
        }
    }
    std::cout<<error<<"\n";
}
