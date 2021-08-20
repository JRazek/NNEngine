//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "Network.h"
#include "layers/ConvolutionLayer.h"
#include "layers/FFLayer.h"
#include "layers/FlatteningLayer.h"

void cn::Network::appendLayer(cn::Layer * layer) {
    //todo validation!
    layers.push_back(layer);
}

void cn::Network::feed(const byte *input) {
    cn::Bitmap<byte> bitmap(inputDataWidth, inputDataHeight, inputDataDepth, input, 0);
    if(layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");
    feed(cn::Utils::normalize(bitmap));
}

cn::Network::~Network() {
    for(auto l : this->layers){
        delete l;
    }
}

void cn::Network::appendConvolutionLayer(int kernelX, int kernelY, int kernelZ, int kernelsCount, const DifferentiableFunction &differentiableFunction, int paddingX,
                                         int paddingY, int strideX, int strideY) {

    ConvolutionLayer *c = new ConvolutionLayer(this->layers.size(), this, kernelX, kernelY, kernelsCount,
                                               differentiableFunction, paddingX, paddingY, strideX, strideY);
    randomInitLayers.push_back(c);
    layers.push_back(c);
}

const std::vector<cn::Layer *> *cn::Network::getLayers() {
    return &layers;
}

cn::Network::Network(int w, int h, int d, int seed)
        : inputDataWidth(w), inputDataHeight(h), inputDataDepth(d), randomEngine(seed) {}

void cn::Network::feed(const cn::Bitmap<float> &bitmap) {
    cn::Bitmap<float> resized = cn::Utils::resize<float>(bitmap, inputDataWidth, inputDataHeight);

    if(layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");
    const Bitmap<float> * input = &resized;
    for(int i = 0; i < layers.size(); i ++){
        auto layer = layers[i];
        layer->run(*input);
        input = &layer->output.value();
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
    for(auto l : randomInitLayers){
        l->randomInit();
    }
}

void cn::Network::appendFFLayer(int neuronsCount, const DifferentiableFunction &differentiableFunction) {
    FFLayer *f = new FFLayer(layers.size(), neuronsCount, differentiableFunction, this);
    randomInitLayers.push_back(f);
    layers.push_back(f);
}

void cn::Network::appendFlatteningLayer() {
    FlatteningLayer *f = new FlatteningLayer(layers.size(), this);
    layers.push_back(f);
}
