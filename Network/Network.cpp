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

void cn::Network::feed(const byte *_input) {
    cn::Bitmap<byte> bitmap(inputSize, _input, 0);
    if(layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");
    feed(cn::Utils::normalize(bitmap));
}

cn::Network::~Network() {
    for(auto l : allocated){
        delete l;
    }
}

void cn::Network::appendConvolutionLayer(int kernelX, int kernelY, int kernelsCount, const DifferentiableFunction &differentiableFunction, int paddingX,
                                         int paddingY, int strideX, int strideY) {

    ConvolutionLayer *c = new ConvolutionLayer(this->layers.size(), *this, kernelX, kernelY, kernelsCount,
                                               differentiableFunction, paddingX, paddingY, strideX, strideY);
    learnableLayers.push_back(c);
    layers.push_back(c);
    allocated.push_back(c);
}
//
//const std::vector<cn::Layer *> *cn::Network::getLayers() {
//    return &layers;
//}

cn::Network::Network(int w, int h, int d, int seed):
        inputSize(w, h, d),
        randomEngine(seed){}

void cn::Network::feed(Bitmap<float> bitmap) {
    if(layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");
    bitmap = Utils::resize(bitmap, inputSize.x, inputSize.y);
    Bitmap<float> *_input = &bitmap;
    input.emplace(*_input);
    outputs.clear();
    for(int i = 0; i < layers.size(); i ++){
        auto layer = layers[i];
        outputs.push_back(layer->run(*_input));
        _input = &outputs.back();
        if(_input->size() != layer->getOutputSize()){
            int here = 1;
        }
    }
}

void cn::Network::feed(const cn::Bitmap<cn::byte> &bitmap) {
    feed(Utils::normalize(bitmap));
}

float cn::Network::getRandom(float low, float high) {
    std::uniform_real_distribution<> dis(low, high);
    return dis(randomEngine);
}

void cn::Network::initRandom() {
    for(auto l : learnableLayers){
        l->randomInit();
    }
}

void cn::Network::appendFFLayer(int neuronsCount, const DifferentiableFunction &differentiableFunction) {
    FFLayer *f = new FFLayer(layers.size(), neuronsCount, differentiableFunction, *this);
    learnableLayers.push_back(f);
    layers.push_back(f);
    allocated.push_back(f);
}

void cn::Network::appendFlatteningLayer() {
    FlatteningLayer *f = new FlatteningLayer(layers.size(), *this);
    layers.push_back(f);
    allocated.push_back(f);
}

void cn::Network::appendBatchNormalizationLayer() {
    BatchNormalizationLayer *b = new BatchNormalizationLayer(layers.size(), *this);
    layers.push_back(b);
    allocated.push_back(b);
}

void cn::Network::appendMaxPoolingLayer(int kernelSizeX, int kernelSizeY) {
    MaxPoolingLayer *m = new MaxPoolingLayer(layers.size(), *this, kernelSizeX, kernelSizeY);
    layers.push_back(m);
    allocated.push_back(m);
}

void cn::Network::ready() {
    outputLayer.emplace(layers.size(), *this);
    layers.push_back(&outputLayer.value());
}

const std::vector<cn::Learnable *> *cn::Network::getLearnables() {
    return &learnableLayers;
}

const cn::Bitmap<float> *cn::Network::getNetworkInput() const {
    return &input.value();
}

void cn::Network::resetMemoization() {
    for(auto l : layers){
        l->resetMemoization();
    }
}

float cn::Network::getChain(int layerID, const Vector3<int> &inputPos) {
    return layers[layerID]->getChain(inputPos);
}

Vector3<int> cn::Network::getOutputSize(int layerID) const {
    return layers[layerID]->getOutputSize();
}

Vector3<int> cn::Network::getInputSize(int layerID) const {
    if(layerID == 0){
        return inputSize;
    }
    return getOutputSize(layerID - 1);
}

const cn::Bitmap<float> *cn::Network::getNetworkOutput() const {
    return &outputs.back();
}

cn::OutputLayer *cn::Network::getOutputLayer() {
    return &outputLayer.value();
}

const cn::Bitmap<float> *cn::Network::getInput(int layerID) const{
    if(layerID == 0)
        return &input.value();
    return getOutput(layerID -1);
}

const cn::Bitmap<float> *cn::Network::getOutput(int layerID) const {
    return &outputs[layerID];
}
