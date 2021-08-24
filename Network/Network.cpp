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
#include "layers/OutputLayer.h"

void cn::Network::feed(const byte *_input) {
    cn::Bitmap<byte> bitmap(inputDataWidth, inputDataHeight, inputDataDepth, _input, 0);
    if(layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");

    input.value() = cn::Utils::normalize(bitmap);
    feed(input.value());
}

cn::Network::~Network() {
    for(auto l : this->layers){
        delete l;
    }
}

void cn::Network::appendConvolutionLayer(int kernelX, int kernelY, int kernelsCount, const DifferentiableFunction &differentiableFunction, int paddingX,
                                         int paddingY, int strideX, int strideY) {
    if(outputLayerAppended){
        delete layers.back();
        layers.pop_back();
    }
    ConvolutionLayer *c = new ConvolutionLayer(this->layers.size(), *this, kernelX, kernelY, kernelsCount,
                                               differentiableFunction, paddingX, paddingY, strideX, strideY);
    learnableLayers.push_back(c);
    layers.push_back(c);
}

const std::vector<cn::Layer *> *cn::Network::getLayers() {
    return &layers;
}

cn::Network::Network(int w, int h, int d, int seed)
        : inputDataWidth(w), inputDataHeight(h), inputDataDepth(d), randomEngine(seed), outputLayerAppended(false) {}

void cn::Network::feed(const cn::Bitmap<float> &bitmap) {
    cn::Bitmap<float> resized = cn::Utils::resize<float>(bitmap, inputDataWidth, inputDataHeight);

    if(layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");
    const Bitmap<float> *_input = &resized;

    if(!outputLayerAppended) {
        outputLayer = new OutputLayer(layers.size(), *this);
        layers.push_back(outputLayer);
        outputLayerAppended = true;
    }
    for(int i = 0; i < layers.size(); i ++){
        Layer *layer = layers[i];
        layer->run(*_input);
        _input = &layer->output.value();
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
}

void cn::Network::appendFlatteningLayer() {
    FlatteningLayer *f = new FlatteningLayer(layers.size(), *this);
    layers.push_back(f);
}

void cn::Network::appendBatchNormalizationLayer() {
    BatchNormalizationLayer *b = new BatchNormalizationLayer(layers.size(), *this);
    layers.push_back(b);
}

void cn::Network::appendMaxPoolingLayer(int kernelSizeX, int kernelSizeY) {
    MaxPoolingLayer *m = new MaxPoolingLayer(layers.size(), *this, kernelSizeX, kernelSizeY);
    layers.push_back(m);
}

cn::Bitmap<float> &cn::Network::getOutput() {
    return layers.back()->output.value();
}
