//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "Network.h"
#include "layers/ConvolutionLayer.h"
#include "layers/FFLayer.h"
#include "layers/FlatteningLayer.h"
#include "layers/BatchNormalizationLayer.h"
#include "layers/MaxPoolingLayer.h"
#include "layers/ActivationLayers/Sigmoid.h"
#include "layers/ActivationLayers/ReLU.h"

void cn::Network::feed(const byte *_input) {
    cn::Bitmap<byte> bitmap(inputSize, _input, 0);
    if(layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");
    feed(cn::Utils::normalize(bitmap));
}

void cn::Network::feed(Bitmap<double> bitmap) {
    if(bitmap.size() != inputSize){
        throw std::logic_error("invalid input size!");
    }
    if(layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");
    const Bitmap<double> *_input = &bitmap;
    input.emplace(*_input);
    outputs.clear();
    for(u_int i = 0; i < layers.size(); i ++){
        auto layer = layers[i];
        outputs.push_back(layer->run(*_input));
        _input = &getOutput(i);
    }
}

void cn::Network::feed(const cn::Bitmap<cn::byte> &bitmap) {
    feed(Utils::normalize(bitmap));
}

double cn::Network::getRandom(double low, double high) {
    std::uniform_real_distribution<> dis(low, high);
    return dis(randomEngine);
}

void cn::Network::initRandom() {
    for(auto l : learnableLayers){
        l->randomInit();
    }
}

void cn::Network::appendConvolutionLayer(int kernelX, int kernelY, int kernelsCount, int strideX, int strideY, int paddingX,
                                         int paddingY) {

    std::unique_ptr<ConvolutionLayer> c = std::make_unique<ConvolutionLayer>(this->layers.size(), *this, kernelX, kernelY, kernelsCount, strideX, strideY, paddingX, paddingY);
    learnableLayers.push_back(c.get());
    layers.push_back(c.get());
    allocated.push_back(std::move(c));
}

void cn::Network::appendFFLayer(int neuronsCount) {
    std::unique_ptr<FFLayer> f = std::make_unique<FFLayer>(layers.size(), neuronsCount, *this);
    learnableLayers.push_back(f.get());
    layers.push_back(f.get());
    allocated.push_back(std::move(f));
}

void cn::Network::appendFlatteningLayer() {
    std::unique_ptr<FlatteningLayer> f = std::make_unique<FlatteningLayer>(layers.size(), *this);
    layers.push_back(f.get());
    allocated.push_back(std::move(f));
}

void cn::Network::appendBatchNormalizationLayer() {
    std::unique_ptr<BatchNormalizationLayer> b = std::make_unique<BatchNormalizationLayer>(layers.size(), *this);
    layers.push_back(b.get());
    allocated.push_back(std::move(b));
}

void cn::Network::appendMaxPoolingLayer(int kernelSizeX, int kernelSizeY) {
    std::unique_ptr<MaxPoolingLayer> m = std::make_unique<MaxPoolingLayer>(layers.size(), *this, kernelSizeX, kernelSizeY);
    layers.push_back(m.get());
    allocated.push_back(std::move(m));
}

void cn::Network::appendReluLayer() {
    std::unique_ptr<ReLU> r = std::make_unique<ReLU>(layers.size(), *this);
    layers.push_back(r.get());
    allocated.push_back(std::move(r));
}

void cn::Network::appendSigmoidLayer() {
    std::unique_ptr<Sigmoid> s = std::make_unique<Sigmoid>(layers.size(), *this);
    layers.push_back(s.get());
    allocated.push_back(std::move(s));
}


void cn::Network::ready() {
    outputLayer.emplace(layers.size(), *this);
    layers.push_back(&outputLayer.value());
}

const std::vector<cn::Learnable *> &cn::Network::getLearnables() const{
    return learnableLayers;
}

void cn::Network::resetMemoization() {
    for(auto l : layers){
        l->resetMemoization();
    }
}

double cn::Network::getChain(int layerID, const Vector3<int> &inputPos) {
    return layers[layerID]->getChain(inputPos);
}

cn::Vector3<int> cn::Network::getOutputSize(int layerID) const {
    return layers[layerID]->getOutputSize();
}

cn::Vector3<int> cn::Network::getInputSize(int layerID) const {
    if(layerID == 0){
        return inputSize;
    }
    return getOutputSize(layerID - 1);
}

const cn::Bitmap<double> &cn::Network::getNetworkOutput() const {
    return outputs.back();
}

cn::OutputLayer &cn::Network::getOutputLayer() {
    return outputLayer.value();
}

const cn::Bitmap<double> &cn::Network::getInput(int layerID) const{
    if(layerID == 0)
        return input.value();
    return getOutput(layerID -1);
}

const cn::Bitmap<double> &cn::Network::getOutput(int layerID) const {
    return outputs[layerID];
}

const std::vector<cn::Layer *> &cn::Network::getLayers() const{
    return layers;
}

cn::JSON cn::Network::jsonEncode() const {
    JSON jsonObject;
    jsonObject["seed"] = seed;
    jsonObject["input_size"] = inputSize.jsonEncode();
    jsonObject["layers"] = std::vector<JSON>();
    for(u_int i = 0; i < layers.size(); i ++){
        jsonObject["layers"].push_back(layers[i]->jsonEncode());
    }
    return jsonObject;
}

cn::Network::Network(cn::Vector3<int> _inputSize, int _seed):
seed(_seed),
inputSize(_inputSize),
randomEngine(_seed)
{}

cn::Network::Network(int w, int h, int d, int _seed):
Network(cn::Vector3<int>(w, h, d), _seed)
{}

cn::Network::Network(const cn::JSON &json): Network(json["input_size"], json["seed"]) {
    JSON _layers = json["layers"];
    for(auto l : _layers){
        allocated.push_back(Layer::fromJSON(*this, l));
        layers.push_back(allocated.back().get());
        if(l["learnable"]){
            learnableLayers.push_back(dynamic_cast<Learnable *>(layers.back()));
        }
    }
}
