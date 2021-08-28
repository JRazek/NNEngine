//
// Created by jrazek on 27.07.2021.
//

#include "FFLayer.h"
#include "../Network.h"
#include "../../Utils/dataStructures/Vector3.h"

cn::FFLayer::FFLayer(int _id, int _neuronsCount, const DifferentiableFunction &_differentiableFunction, Network &_network) :
        Learnable(_id, _network, _neuronsCount),
        differentiableFunction(_differentiableFunction),
        biases(_neuronsCount){
    if(__id == 0){
        throw std::logic_error("FFLayer must not be the first layer in the network!");
    }else{
        const Bitmap<float> &prev = network->getInput(__id);
        if(prev.w() < 1 || prev.h() != 1 || prev.d() != 1){
            throw std::logic_error("There must be a vector output layer before FFLayer!");
        }
        weights = std::vector<float>(neuronsCount * prev.w());
    }
    output.emplace(Bitmap<float>(neuronsCount, 1, 1));
}

void cn::FFLayer::run(const Bitmap<float> &input) {
    _input = &input;
    if(input.w() < 1 || input.h() != 1 || input.d() != 1){
        throw std::logic_error("input bitmap to ff layer must be a normalized vector type!");
    }
    netSums.emplace(Bitmap<float>(neuronsCount, 1, 1));
    int weightsPerNeuron = weightsCount() / neuronsCount;
    for(int n = 0; n < neuronsCount; n ++){
        float sum = biases[n];
        for(int i = 0; i < input.w(); i ++){
            sum += getWeight(n * weightsPerNeuron + i) * input.getCell(i, 0, 0);
        }
        netSums.value().setCell(n, 0, 0, sum);
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

float cn::FFLayer::getChain(const Vector3<int> &inputPos) {
    if(inputPos.x < 0 || inputPos.y != 0 || inputPos.z != 0){
        throw std::logic_error("wrong chain request!");
    }
    if(getMemoState(inputPos)){
        return getMemo(inputPos);
    }
    int weightsPerNeuron = weights.size() / neuronsCount;
    float sum = 0;
    for(int i = 0; i < neuronsCount; i ++){
        int weightID = weightsPerNeuron * i + inputPos.x;
        sum += weights[weightID] * differentiableFunction.derive(_input->getCell(inputPos)) * network->getChain(__id + 1, {i, 0, 0});
    }
    setMemo(inputPos, sum);
    return sum;
}

float cn::FFLayer::diffWeight(int weightID) {
    int neuronID = weightID / (weightsCount()/neuronsCount);
    int weightsPerNeuron = weightsCount() / neuronsCount;
    return _input->getCell(weightID % weightsPerNeuron, 0, 0)
            * differentiableFunction.derive(netSums->getCell(neuronID, 0, 0))
            * network->getChain(__id + 1, {neuronID, 0, 0});
}

int cn::FFLayer::weightsCount() const {
    return weights.size();
}

std::vector<float> cn::FFLayer::getGradient() {
    std::vector<float> gradient(weightsCount());
    for(int i = 0; i < weightsCount(); i ++){
        gradient[i] = diffWeight(i);
    }
    return gradient;
}

void cn::FFLayer::setWeight(int weightID, float value) {
    weights[weightID] = value;
}

float cn::FFLayer::getWeight(int weightID) const {
    return weights[weightID];
}
