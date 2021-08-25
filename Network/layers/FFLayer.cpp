//
// Created by jrazek on 27.07.2021.
//

#include "FFLayer.h"
#include "../Network.h"
#include "../../Utils/dataStructures/Vector3.h"

cn::FFLayer::FFLayer(int _id, int _neuronsCount, const DifferentiableFunction &_differentiableFunction, Network &_network) :
        Learnable(_id, _network),
        neuronsCount(_neuronsCount),
        differentiableFunction(_differentiableFunction),
        biases(_neuronsCount){
    if(__id == 0){
        throw std::logic_error("FFLayer must not be the first layer in the network!");
    }else{
        const Bitmap<float> *prev = network->getLayers()->at(__id - 1)->getOutput();
        if(prev->w() < 1 || prev->h() != 1 || prev->d() != 1){
            throw std::logic_error("There must be a vector output layer before FFLayer!");
        }
        weights = std::vector<float>(neuronsCount * prev->w());
    }
    output.emplace(Bitmap<float>(neuronsCount, 1, 1));
}

void cn::FFLayer::run(const Bitmap<float> &input) {
    _input = &input;
    if(input.w() < 1 || input.h() != 1 || input.d() != 1){
        throw std::logic_error("input bitmap to ff layer must be a normalized vector type!");
    }
    netSums.emplace(Bitmap<float>(neuronsCount, 1, 1));
    for(int n = 0; n < neuronsCount; n ++){
        float sum = biases[n];
        for(int i = 0; i < input.w(); i ++){
            sum += getWeight(n, i) * input.getCell(i, 0, 0);
        }
        netSums.value().setCell(n, 1, 1, sum);
        output->setCell(n, 0, 0, differentiableFunction.func(sum));
    }
}

void cn::FFLayer::randomInit() {
    for(auto &w : weights){
        w = network->getRandom(-1, 1);
    }
    for(auto &b : biases){
        b = network->getRandom(-5, 5);
    }
}

float cn::FFLayer::getWeight(int neuron, int weightID) {
    int perNeuron = weights.size() / neuronsCount;
    return weights[perNeuron * neuron + weightID];
}

float cn::FFLayer::getChain(const Vector3<int> &input) {
    if(input.x < 0 || input.y != 0 || input.z != 0){
        throw std::logic_error("wrong chain request!");
    }
    int weightsPerNeuron = weights.size() / neuronsCount;
    float sum = 0;
    for(int i = 0; i < neuronsCount; i ++){
        int weightID = weightsPerNeuron * i + input.x;
        sum += weights[weightID] * differentiableFunction.derive(_input->getCell(input)) * network->getLayers()->at(__id + 1)->getChain({i, 0, 0});
    }
    return sum;
}

float cn::FFLayer::diffWeight(int neuronID, int weightID) {
    return network->getLayers()->at(__id - 1)->getOutput()->getCell(weightID, 0, 0)
            * differentiableFunction.derive(netSums->getCell(neuronID, 0, 0))
            * network->getLayers()->at(__id + 1)->getChain({neuronID, 0, 0});
}
