//
// Created by user on 21.08.2021.
//

#include "Backpropagation.h"
#include "../Utils/Bitmap.h"
#include "../Network/Network.h"
cn::Backpropagation::Backpropagation(Network &_network, float _learningRate) :
        network(_network),
        learningRate(_learningRate)
        {}

void cn::Backpropagation::propagate(const cn::Bitmap<float> &target) {
    network.resetMemoization();
    OutputLayer *layer = network.getOutputLayer();
    layer->setTarget(&target);
    const Bitmap<float> &output = *network.getOutput(layer->id());
    if(output.size() != target.size()){
        throw std::logic_error("Backpropagation, invalid target!");
    }
    float error = 0;
    for(int i = 0; i < target.w(); i ++){
        error += std::pow(target.getCell(i, 0, 0) - output.getCell(i, 0, 0), 2);
    }
//    for(Learnable *learnable : *network.getLearnables()){
//        std::vector<float> layerGradient = learnable->getGradient();
//        for(int i = 0; i < layerGradient.size(); i ++){
//            learnable->setWeight(i, learnable->getWeight(i) - learningRate * layerGradient[i]);
//        }
//    }
    std::cout<<error<<"\n";
}
