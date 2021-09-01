//
// Created by jrazek on 27.07.2021.
//

#include "FFLayer.h"
#include "../Network.h"

cn::FFLayer::FFLayer(int _id, int _neuronsCount, const DifferentiableFunction &_differentiableFunction, Network &_network) :
        Learnable(_id, _network, _neuronsCount),
        biases(_neuronsCount),
        differentiableFunction(_differentiableFunction),
        beforeActivation(_neuronsCount){

    if(inputSize.x < 1 || inputSize.y != 1 || inputSize.z != 1){
        throw std::logic_error("There must be a vector output layer before FFLayer!");
    }
    weights = std::vector<double>(neuronsCount * inputSize.x);
    outputSize = Vector3<int> (neuronsCount, 1, 1);
}

cn::Bitmap<double> cn::FFLayer::run(const Bitmap<double> &input) {
    if(input.w() < 1 || input.h() != 1 || input.d() != 1){
        throw std::logic_error("input bitmap to ff layer must be a normalized vector type!");
    }
    int weightsPerNeuron = weightsCount() / neuronsCount;

    Bitmap<double> result(neuronsCount, 1, 1);
    for(int i = 0; i < neuronsCount; i ++){
        double sum = 0;
        for(int j = 0; j < weightsPerNeuron; j ++){
            sum += input.getCell(j, 0, 0) * weights.at(i * weightsPerNeuron + j);
        }
        beforeActivation[i] = sum;
        double activated = differentiableFunction.func(sum);
        result.setCell(i, 0, 0, activated);
    }
    return result;
}

void cn::FFLayer::randomInit() {
    for(auto &w : weights){
        w = network->getRandom(-10, 10)* std::sqrt(2.f/inputSize.x);
    }
    for(auto &b : biases){
        b = network->getRandom(0, 0);
    }
}

double cn::FFLayer::getChain(const Vector3<int> &inputPos) {
    if(inputPos.x < 0 || inputPos.y != 0 || inputPos.z != 0){
        throw std::logic_error("wrong chain request!");
    }
    if(getMemoState(inputPos)){
        return getMemo(inputPos);
    }
    int weightsPerNeuron = inputSize.x;

    double res = 0;
    for(int i = 0; i < neuronsCount; i ++){
        res += weights.at(i * weightsPerNeuron  + inputPos.x) * differentiableFunction.derive(beforeActivation.at(i)) * network->getChain(__id + 1, {i, 0, 0});
    }

    setMemo(inputPos, res);
    return res;
}

double cn::FFLayer::diffWeight(int weightID) {
    int neuron = weightID / inputSize.x;
    const Bitmap<double> *input = network->getInput(__id);
    return input->getCell(weightID % inputSize.x, 0, 0) * differentiableFunction.derive(beforeActivation.at(neuron)) * network->getChain(__id + 1, {neuron, 0, 0});
}

int cn::FFLayer::weightsCount() const {
    return weights.size();
}

std::vector<double> cn::FFLayer::getWeightsGradient() {
    std::vector<double> gradient(weightsCount());
    for(int i = 0; i < weightsCount(); i ++){
        gradient[i] = diffWeight(i);
    }
    return gradient;
}

void cn::FFLayer::setWeight(int weightID, double value) {
    weights[weightID] = value;
}

double cn::FFLayer::getWeight(int weightID) const {
    return weights[weightID];
}

std::vector<double> cn::FFLayer::getBiasesGradient() {
    std::vector<double> gradient(neuronsCount);
    for(int i = 0; i < neuronsCount; i++){
        gradient[i] = diffBias(i);
    }
    return gradient;
}

double cn::FFLayer::diffBias(int neuronID) {
    return differentiableFunction.derive(beforeActivation[neuronID]) * network->getChain(__id + 1, {neuronID, 0, 0});
}

void cn::FFLayer::setBias(int neuronID, double value) {
    biases[neuronID] = value;
}

double cn::FFLayer::getBias(int neuronID) const {
    return biases[neuronID];
}

int cn::FFLayer::biasesCount() const {
    return neuronsCount;
}
