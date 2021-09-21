//
// Created by user on 05.09.2021.
//

#include "Optimizer.h"

cn::Optimizer::Optimizer(Network &_network, double _learningRate) :
network(_network),
iteration(0),
learningRate(_learningRate) {}

double cn::Optimizer::getError(const cn::Bitmap<double> &target) const {
    double error = 0;
    OutputLayer &layer = network.getOutputLayer();
    const Bitmap<double> &output = *network.getOutput(layer.id());
    for(int i = 0; i < target.w(); i ++){
        error += 0.5*std::pow(target.getCell(i, 0, 0) - output.getCell(i, 0, 0), 2);
    }
    return error;
}

const std::vector<cn::Layer *> &cn::Optimizer::getNetworkLayers() {
    return network.layers;
}

const std::vector<cn::Learnable *> &cn::Optimizer::getLearnables() {
    return network.learnableLayers;
}
