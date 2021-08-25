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
    cn::Bitmap<byte> bitmap(inputDataWidth, inputDataHeight, inputDataDepth, _input, 0);
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

const std::vector<cn::Layer *> *cn::Network::getLayers() {
    return &layers;
}

cn::Network::Network(int w, int h, int d, int seed):
        //outputLayer(-1, *this),
        inputDataWidth(w),
        inputDataHeight(h),
        inputDataDepth(d),
        randomEngine(seed)
        {}

void cn::Network::feed(const cn::Bitmap<float> &bitmap) {
    if(layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");
    input.emplace(cn::Utils::resize<float>(bitmap, inputDataWidth, inputDataHeight));

    const Bitmap<float> *_input = &input.value();
    for(int i = 0; i < layers.size(); i ++){
        auto layer = layers[i];
        layer->run(*_input);
        _input = layer->getOutput();
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
    if(!outputLayer.has_value()) {
        outputLayer.emplace(layers.size(), *this);
        layers.push_back(&outputLayer.value());
    }
}

cn::OutputLayer *cn::Network::getOutputLayer() {
    return &outputLayer.value();
}

const std::vector<cn::Learnable *> *cn::Network::getLearnables() {
    return &learnableLayers;
}

