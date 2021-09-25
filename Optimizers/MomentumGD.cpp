//
// Created by user on 05.09.2021.
//

#include "MomentumGD.h"

cn::MomentumGD::MomentumGD(cn::Network &_network, float _theta, double _learningRate):
Optimizer(_network, _learningRate),
theta(_theta)
{}

void cn::MomentumGD::propagate(const cn::Tensor<double> &target, bool CUDAAccelerate) {
    network.resetMemoization();
    const std::vector<cn::Learnable *> &learnables = getLearnables();
    const std::vector<cn::Layer *> &layers = getNetworkLayers();

    if(!iteration){
        emaWeightsMemo = std::vector<std::vector<float>>(learnables.size(), std::vector<float>());
        emaBiasesMemo = std::vector<std::vector<float>>(learnables.size(), std::vector<float>());
        for(u_int i = 0; i < learnables.size(); i ++){
            emaWeightsMemo[i] = std::vector<float>(learnables.at(i)->weightsCount(), 0);
            emaBiasesMemo[i] = std::vector<float>(learnables.at(i)->biasesCount(), 0);
        }
    }
    OutputLayer &layer = network.getOutputLayer();
    layer.setTarget(&target);
    const Tensor<double> &output = *network.getOutput(layer.id());
    if(output.size() != target.size()){
        throw std::logic_error("MBGD, invalid target!");
    }

    if(CUDAAccelerate){
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
            emaWeightsMemo[k][i] = theta * emaWeightsMemo[k][i] + (1.f - theta) * (learnable->getWeight(i) - learningRate * weightsGradient[i]);
            learnable->setWeight(i, emaWeightsMemo[k][i]);
        }
        for(u_int i = 0; i < biasesGradient.size(); i ++){
            emaBiasesMemo[k][i] = theta * emaBiasesMemo[k][i] + (1.f - theta) * (learnable->getBias(i) - learningRate * biasesGradient[i]);
            learnable->setBias(i, emaBiasesMemo[k][i]);
        }
    }
    iteration++;
}
