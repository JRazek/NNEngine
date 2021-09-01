//
// Created by user on 21.08.2021.
//

#include "Backpropagation.h"
#include "../Utils/dataStructures/Bitmap.h"
#include "../Network/Network.h"
cn::Backpropagation::Backpropagation(Network &_network, double _learningRate, int _miniBatchSize) :
        network(_network),
        learningRate(_learningRate),
        miniBatchSize(_miniBatchSize),
        iteration(0){
    if(miniBatchSize <= 0){
        throw std::logic_error("mini-batch size must be a positive integer!");
    }
}

void cn::Backpropagation::propagate(const cn::Bitmap<double> &target) {
    network.resetMemoization();
    if(!(iteration % miniBatchSize)){
        memorizedWeights.clear();
        memorizedBiases.clear();
        memorizedWeights.resize(network.getLearnables()->size(), std::vector<double>());
        memorizedBiases.resize(network.getLearnables()->size(), std::vector<double>());
        for(int i = 0; i < memorizedWeights.size(); i ++){
            memorizedWeights[i].resize(network.getLearnables()->at(i)->weightsCount(), 0);
        }
        for(int i = 0; i < memorizedBiases.size(); i ++){
            memorizedBiases[i].resize(network.getLearnables()->at(i)->biasesCount(), 0);
        }
    }
    OutputLayer *layer = network.getOutputLayer();
    layer->setTarget(&target);
    const Bitmap<double> &output = *network.getOutput(layer->id());
    if(output.size() != target.size()){
        throw std::logic_error("Backpropagation, invalid target!");
    }
    for(int k = 0; k < network.getLearnables()->size(); k ++){
        Learnable *learnable = network.getLearnables()->at(k);
        std::vector<double> weightsGradient = learnable->getWeightsGradient();
        std::vector<double> biasesGradient = learnable->getBiasesGradient();
        for(int i = 0; i < weightsGradient.size(); i ++){
            memorizedWeights[k][i] += learnable->getWeight(i) - learningRate * weightsGradient[i];
        }
        for(int i = 0; i < biasesGradient.size(); i ++){
            memorizedBiases[k][i] += learnable->getBias(i) - learningRate * biasesGradient[i];
        }
    }

    if(!((iteration + 1) % miniBatchSize)){
        for(int k = 0; k < network.getLearnables()->size(); k ++) {
            Learnable *learnable = network.getLearnables()->at(k);
            std::vector<double> &layerWeightsSum = memorizedWeights[k];
            std::vector<double> &layerBiasesSum = memorizedBiases[k];

            for(int i = 0; i < layerWeightsSum.size(); i ++) {
                double resultGradient = layerWeightsSum[i] / miniBatchSize;
                learnable->setWeight(i, resultGradient);
            }
            for(int i = 0; i < layerBiasesSum.size(); i ++) {
                double resultGradient = layerBiasesSum[i] / miniBatchSize;
                learnable->setBias(i, resultGradient);
            }
        }
    }

    iteration++;
}

double cn::Backpropagation::getError(const cn::Bitmap<double> &target) const {
    double error = 0;
    OutputLayer *layer = network.getOutputLayer();
    const Bitmap<double> &output = *network.getOutput(layer->id());
    for(int i = 0; i < target.w(); i ++){
        error += 0.5*std::pow(target.getCell(i, 0, 0) - output.getCell(i, 0, 0), 2);
    }
    return error;
}
