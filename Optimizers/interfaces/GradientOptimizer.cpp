//
// Created by user on 05.09.2021.
//

#include "GradientOptimizer.h"

cn::GradientOptimizer::Optimizer(Network &_network, double _learningRate) :
network(&_network),
iteration(0),
learningRate(_learningRate) {}

double cn::GradientOptimizer::getError(const cn::Tensor<double> &target) const {
    double error = 0;
    OutputLayer &layer = network->getOutputLayer();
    const Tensor<double> &output = network->getOutput(network->layersCount() - 1, layer.getTime() - 1);
    for(int i = 0; i < target.w(); i ++){
        error += 0.5*std::pow(target.getCell(i, 0, 0) - output.getCell(i, 0, 0), 2);
    }
    return error;
}

const std::vector<std::unique_ptr<cn::Layer>> & cn::GradientOptimizer::getNetworkLayers() const {
    return network->layers;
}

const std::vector<cn::Learnable *> &cn::GradientOptimizer::getLearnables() const {
    return network->learnableLayers;
}
