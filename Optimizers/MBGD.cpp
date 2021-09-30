//
// Created by user on 21.08.2021.
//

#include "MBGD.h"
#include "../Utils/dataStructures/Tensor.h"

cn::MBGD::MBGD(Network &_network, double _learningRate, int _miniBatchSize) :
        Optimizer(_network, _learningRate),
        miniBatchSize(_miniBatchSize){
    if(miniBatchSize <= 0){
        throw std::logic_error("mini-batch size must be a positive integer!");
    }
}

void cn::MBGD::propagate(const Tensor<double> &target) {
    const std::vector<cn::Learnable *> &learnables = getLearnables();
    const std::vector<cn::Layer *> &layers = getNetworkLayers();

    if(!(iteration % miniBatchSize)){
        memorizedWeights.clear();
        memorizedBiases.clear();
        memorizedWeights.resize(learnables.size(), std::vector<double>());
        memorizedBiases.resize(learnables.size(), std::vector<double>());
        for(u_int i = 0; i < memorizedWeights.size(); i ++){
            memorizedWeights[i].resize(learnables.at(i)->weightsCount(), 0);
        }
        for(u_int i = 0; i < memorizedBiases.size(); i ++){
            memorizedBiases[i].resize(learnables.at(i)->biasesCount(), 0);
        }
    }
    OutputLayer &layer = network.getOutputLayer();
    layer.setTarget(&target);
    //todo time set
    const Tensor<double> &output = network.getOutput(layer.id(), 0);
    if(output.size() != target.size()){
        throw std::logic_error("MBGD, invalid target!");
    }

    if(network.isCudaAccelerate()){
        for(auto it = layers.rbegin(); it != layers.rend(); ++it){
            (*it)->CUDAAutoGrad();
            std::cout<<"";
        }
    }

    for(u_int k = 0; k < learnables.size(); k ++){
        Learnable *learnable = learnables.at(k);
        std::vector<double> weightsGradient = learnable->getWeightsGradient();
        std::vector<double> biasesGradient = learnable->getBiasesGradient();
        for(u_int i = 0; i < weightsGradient.size(); i ++){
            memorizedWeights[k][i] += learnable->getWeight(i) - learningRate * weightsGradient[i];
        }
        for(u_int i = 0; i < biasesGradient.size(); i ++){
            memorizedBiases[k][i] += learnable->getBias(i) - learningRate * biasesGradient[i];
        }
    }

    if(!((iteration + 1) % miniBatchSize)){
        for(u_int k = 0; k < learnables.size(); k ++) {
            Learnable *learnable = learnables.at(k);
            std::vector<double> &layerWeightsSum = memorizedWeights[k];
            std::vector<double> &layerBiasesSum = memorizedBiases[k];

            for(u_int i = 0; i < layerWeightsSum.size(); i ++) {
                double resultGradient = layerWeightsSum[i] / miniBatchSize;
                learnable->setWeight(i, resultGradient);
            }
            for(u_int i = 0; i < layerBiasesSum.size(); i ++) {
                double resultGradient = layerBiasesSum[i] / miniBatchSize;
                learnable->setBias(i, resultGradient);
            }
        }
    }

    iteration++;
}