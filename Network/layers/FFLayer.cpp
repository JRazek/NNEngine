//
// Created by jrazek on 27.07.2021.
//

#include "FFLayer.h"
#include "../Network.h"

cn::FFLayer::FFLayer(int _id, int _neuronsCount, const DifferentiableFunction &_differentiableFunction, Network &_network) :
        Learnable(_id, _network, _neuronsCount),
        differentiableFunction(_differentiableFunction),
        biases(_neuronsCount),
        beforeActivation(_neuronsCount){
    if(__id == 0){
        throw std::logic_error("FFLayer must not be the first layer in the network!");
    }else{
        if(inputSize.x < 1 || inputSize.y != 1 || inputSize.z != 1){
            throw std::logic_error("There must be a vector output layer before FFLayer!");
        }
        weights = std::vector<float>(neuronsCount * inputSize.x);
    }
    outputSize = Vector3<int> (neuronsCount, 1, 1);
}

cn::Bitmap<float> cn::FFLayer::run(const Bitmap<float> &input) {
    if(input.w() < 1 || input.h() != 1 || input.d() != 1){
        throw std::logic_error("input bitmap to ff layer must be a normalized vector type!");
    }
    int weightsPerNeuron = weightsCount() / neuronsCount;

    Bitmap<float> result(outputSize);
    for(int n = 0; n < neuronsCount; n ++){
        float sum = biases[n];
        for(int i = 0; i < input.w(); i ++){
            sum += getWeight(n * weightsPerNeuron + i) * input.getCell(i, 0, 0);
        }
        beforeActivation[n] = sum;
        result.setCell(n, 0, 0, differentiableFunction.func(sum));
    }
    return result;
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
        sum += weights.at(weightID) * differentiableFunction.derive(beforeActivation.at(i)) * network->getChain(__id + 1, {i, 0, 0});
    }
    setMemo(inputPos, sum);
    return sum;
}

float cn::FFLayer::diffWeight(int weightID) {
    int weightsPerNeuron = weightsCount() / neuronsCount;
    int neuronID = weightID / weightsPerNeuron;
    float res = network->getInput(__id)->getCell(weightID % weightsPerNeuron, 0, 0)
            * differentiableFunction.derive(beforeActivation.at(neuronID));
    return res * network->getChain(__id + 1, {neuronID, 0, 0});
}

int cn::FFLayer::weightsCount() const {
    return weights.size();
}

std::vector<float> cn::FFLayer::getWeightsGradient() {
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

std::vector<float> cn::FFLayer::getBiasesGradient() {
    std::vector<float> gradient(neuronsCount);
    for(int i = 0; i < neuronsCount; i++){
        gradient[i] = diffBias(i);
    }
    return gradient;
}

float cn::FFLayer::diffBias(int neuronID) {
    return differentiableFunction.derive(beforeActivation[neuronID]) * network->getChain(__id + 1, {neuronID, 0, 0});
}

void cn::FFLayer::setBias(int neuronID, float value) {
    biases[neuronID] = value;
}

float cn::FFLayer::getBias(int neuronID) const {
    return biases[neuronID];
}

int cn::FFLayer::biasesCount() const {
    return neuronsCount;
}
