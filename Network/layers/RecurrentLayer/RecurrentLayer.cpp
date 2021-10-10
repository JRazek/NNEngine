//
// Created by jrazek on 24.09.2021.
//
#include "../../Network.h"
#include "RecurrentLayer.h"
#include "../InputLayer/InputLayer.h"
#include "RecurrentOutputLayer/RecurrentOutputLayer.h"
#include "../ActivationLayers/Sigmoid/Sigmoid.h"

std::unique_ptr<cn::Layer> cn::RecurrentLayer::getCopyAsUniquePtr() const noexcept{
    return std::make_unique<RecurrentLayer>(*this);
}

cn::RecurrentLayer::RecurrentLayer(const Vector3<int> &_inputSize, std::vector<std::unique_ptr<Layer>> &&layers) :
Learnable(_inputSize),
internalLayers(std::move(layers)),
identity(inputSize){
    std::fill(identity.data(), identity.data() + identity.size().multiplyContent(), 0);
    internalLayers.push_back(std::make_unique<InputLayer>(InputLayer(inputSize)));
}

void cn::RecurrentLayer::CPURun(const cn::Tensor<double> &_input) {
    for(auto &l : internalLayers)
        l->incTime();
    Tensor<double> res = Utils::elementWiseSum(_input, _time == 0 ? identity : output[getTime() - 1]);
    const Tensor<double> *input = &res;
    for(u_int i = 0; i < internalLayers.size(); i ++){
        internalLayers[i]->CPURun(*input);
        input = &internalLayers[i]->getOutput(getTime());
    }
    for(auto &l : internalLayers)
        l->incTime();

    output.push_back(*input);

    //todo check this!!!!!!
}

double cn::RecurrentLayer::getChain(const Vector4<int> &inputPos) {
    return internalLayers[0]->getChain(inputPos);
}

cn::JSON cn::RecurrentLayer::jsonEncode() const {
    JSON structure;
    structure["input_size"] = inputSize.jsonEncode();
    structure["type"] = "rcl";
    structure["internal_layers"] = std::vector<JSON>();
    for(auto &l : internalLayers){
        structure["internal_layers"].push_back(l->jsonEncode());
    }
    return structure;
}

cn::RecurrentLayer::RecurrentLayer(const cn::JSON &json) :
RecurrentLayer(Vector3<int>(json.at("input_size"))) {
    for(auto &layerJSON : json.at("internal_layers")){
        std::unique_ptr<Layer> layer = Layer::fromJSON(layerJSON);
        internalLayers.push_back(std::move(layer));
    }
}

cn::RecurrentLayer::RecurrentLayer(const cn::RecurrentLayer &recurrentLayer): Learnable(recurrentLayer) {
    for(u_int i = 0; i < recurrentLayer.internalLayers.size() - recurrentLayer._ready; i ++){
        internalLayers.push_back(recurrentLayer.internalLayers[i]->getCopyAsUniquePtr());
    }
    if(recurrentLayer._ready){
        ready();
    }
}

cn::RecurrentLayer::RecurrentLayer(const Vector3<int> &_inputSize) : Learnable(_inputSize), identity(inputSize)  {
    std::fill(identity.data(), identity.data() + identity.size().multiplyContent(), 0);
}

void cn::RecurrentLayer::ready() {
    if(!_ready) {
        internalLayers.push_back(std::make_unique<RecurrentOutputLayer>(RecurrentOutputLayer(internalLayers.back()->getOutputSize(), *this)));
        Network::linkLayers(internalLayers);
        outputSize = inputSize;
        _ready = true;
    }
}

double cn::RecurrentLayer::getChainFromChild(const cn::Vector4<int> &inputPos) {
    return nextLayer->getChain(inputPos);
}

void cn::RecurrentLayer::randomInit(std::default_random_engine &randomEngine) {

}

double cn::RecurrentLayer::diffWeight(int weightID) {
    return 0;
}

double cn::RecurrentLayer::diffBias(int neuronID) {
    return 0;
}

std::vector<double> cn::RecurrentLayer::getWeightsGradient() {
    return std::vector<double>();
}

std::vector<double> cn::RecurrentLayer::getBiasesGradient() {
    return std::vector<double>();
}

double cn::RecurrentLayer::getBias(int neuronID) const {
    return 0;
}

void cn::RecurrentLayer::setBias(int neuronID, double value) {

}

int cn::RecurrentLayer::weightsCount() const {
    return 0;
}

int cn::RecurrentLayer::biasesCount() const {
    return 0;
}

void cn::RecurrentLayer::setWeight(int weightID, double value) {

}

double cn::RecurrentLayer::getWeight(int weightID) const {
    return 0;
}

void cn::RecurrentLayer::appendSigmoidLayer() {
    std::unique_ptr<Sigmoid> s = std::make_unique<Sigmoid>(internalLayers.back()->getOutputSize());
    internalLayers.push_back(std::move(s));
}

