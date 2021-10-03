//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "Network.h"
#include "layers/ConvolutionLayer/ConvolutionLayer.h"
#include "layers/FFLayer/FFLayer.h"
#include "layers/FlatteningLayer/FlatteningLayer.h"
#include "layers/BatchNormalizationLayer/BatchNormalizationLayer.h"
#include "layers/MaxPoolingLayer/MaxPoolingLayer.h"
#include "layers/InputLayer/InputLayer.h"
#include "layers/ActivationLayers/Sigmoid/Sigmoid.h"
#include "layers/ActivationLayers/ReLU/ReLU.h"
#include "layers/RecurrentLayer/RecurrentLayer.h"

void cn::Network::feed(Tensor<double> bitmap) {
#ifndef NNL_WITH_CUDA
    if(CUDAAccelerate){
        throw std::logic_error("CUDA IS NOT AVAILABLE! set -DCOMPILE_WITH_CUDA=ON during compilation.");
    }
#endif
    if(bitmap.size() != inputSize){
        throw std::logic_error("invalid input size!");
    }
    if(layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");

    const Tensor<double> *_input = &bitmap;
    for(u_int i = 0; i < layers.size(); i ++){

        auto layer = layers[i];
        if(!CUDAAccelerate)
            layer->CPURun(*_input);
        else
            layer->CUDARun(*_input);
        _input = &getOutput(i, layer->getTime());
    }

    for(auto l : layers)
        l->incTime();
}

void cn::Network::feed(const cn::Tensor<cn::byte> &bitmap) {
    feed(Utils::normalize(bitmap));
}

void cn::Network::initRandom() {
    for(auto l : learnableLayers){
        l->randomInit(randomEngine);
    }
}

void cn::Network::appendConvolutionLayer(Vector2<int> kernelSize, int kernelsCount, Vector2<int> stride,
                                         Vector2<int> padding) {

    int id = this->layers.size();
    std::unique_ptr<ConvolutionLayer> c = std::make_unique<ConvolutionLayer>(id, getInputSize(id), kernelSize, kernelsCount, stride, padding);
    learnableLayers.push_back(c.get());
    layers.push_back(c.get());
    allocated.push_back(std::move(c));
}

void cn::Network::appendFFLayer(int neuronsCount) {
    int id = this->layers.size();
    std::unique_ptr<FFLayer> f = std::make_unique<FFLayer>(id, getInputSize(id), neuronsCount);
    learnableLayers.push_back(f.get());
    layers.push_back(f.get());
    allocated.push_back(std::move(f));
}

void cn::Network::appendFlatteningLayer() {
    int id = this->layers.size();
    std::unique_ptr<FlatteningLayer> f = std::make_unique<FlatteningLayer>(id, getInputSize(id));
    layers.push_back(f.get());
    allocated.push_back(std::move(f));
}

void cn::Network::appendBatchNormalizationLayer() {
    int id = this->layers.size();
    std::unique_ptr<BatchNormalizationLayer> b = std::make_unique<BatchNormalizationLayer>(id, getInputSize(id));
    layers.push_back(b.get());
    allocated.push_back(std::move(b));
}

void cn::Network::appendMaxPoolingLayer(Vector2<int> kernelSize) {
    int id = this->layers.size();
    std::unique_ptr<MaxPoolingLayer> m = std::make_unique<MaxPoolingLayer>(id, getInputSize(id), kernelSize);
    layers.push_back(m.get());
    allocated.push_back(std::move(m));
}

void cn::Network::appendReLULayer() {
    int id = this->layers.size();
    std::unique_ptr<ReLU> r = std::make_unique<ReLU>(id, getInputSize(id));
    layers.push_back(r.get());
    allocated.push_back(std::move(r));
}

void cn::Network::appendSigmoidLayer() {
    int id = this->layers.size();
    std::unique_ptr<Sigmoid> s = std::make_unique<Sigmoid>(id, getInputSize(id));
    layers.push_back(s.get());
    allocated.push_back(std::move(s));
}


void cn::Network::ready() {
    int id = this->layers.size();
    if(!outputLayer) {
        std::unique_ptr<OutputLayer> _outputLayer = std::make_unique<OutputLayer>(id, getInputSize(id));
        outputLayer = _outputLayer.get();
        layers.push_back(outputLayer);
        allocated.push_back(std::move(_outputLayer));
    }
    linkLayers();
    resetState();
}

void cn::Network::resetState() {
    for(auto l : layers){
        l->resetState();
    }
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

const cn::Tensor<double> &cn::Network::getNetworkOutput(int time) const {
    return layers.back()->getOutput(time);
}

cn::OutputLayer &cn::Network::getOutputLayer() {
    return *outputLayer;
}

const cn::Tensor<double> &cn::Network::getInput(int layerID, int time) const{
    if(layerID == 0)
        return inputLayer->getInput(time);

    return getOutput(layerID - 1, time);
}

const cn::Tensor<double> & cn::Network::getOutput(int layerID, int time) const {
    return layers[layerID]->getOutput(time);
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


cn::Network::Network(cn::Vector3<int> _inputSize, int _seed, bool _CUDAAccelerate):
seed(_seed),
inputSize(_inputSize),
randomEngine(_seed){
    std::unique_ptr<InputLayer> _inputLayer = std::make_unique<InputLayer>(0, inputSize);
    layers.push_back(_inputLayer.get());
    allocated.push_back(std::move(_inputLayer));
    inputLayer = _inputLayer.get();
    CUDAAccelerate = _CUDAAccelerate;
}

cn::Network::Network(int w, int h, int d, int _seed, bool _CUDAAccelerate):
Network(cn::Vector3<int>(w, h, d), _seed, _CUDAAccelerate)
{}

cn::Network::Network(const cn::JSON &json): seed(json.at("seed")), inputSize(json.at("input_size")) {
    try {
        JSON _layers = json.at("layers");
        for (auto l : _layers) {
            allocated.push_back(Layer::fromJSON(l));
            layers.push_back(allocated.back().get());
            if (l.contains("learnable") && l.at("learnable")) {
                learnableLayers.push_back(dynamic_cast<Learnable *>(layers.back()));
            }
            if(l.at("type") == "il"){
                inputLayer = dynamic_cast<InputLayer *>(layers.back());
            }
            if(l.at("type") == "ol"){
                outputLayer = dynamic_cast<OutputLayer *>(layers.back());
            }
        }
        linkLayers();
    }catch(std::exception &e){
        std::cout<<e.what();
    }
}

cn::Network::Network(cn::Network &&network):
seed(network.seed),
inputSize(network.inputSize),
randomEngine(std::move(network.randomEngine)),
allocated(std::move(network.allocated)),
learnableLayers(std::move(network.learnableLayers)),
layers(std::move(network.layers)),
outputLayer(network.outputLayer)
{}

cn::Network &cn::Network::operator=(cn::Network &&network) {
    seed = network.seed;
    inputSize = network.inputSize;
    randomEngine = std::move(network.randomEngine);
    allocated = std::move(network.allocated);
    layers = std::move(network.layers);
    learnableLayers = std::move(network.learnableLayers);
    inputLayer = network.inputLayer;
    outputLayer = network.outputLayer;

    return *this;
}

void cn::Network::linkLayers() {
    for(u_int i = 0; i < layers.size(); i ++){
        if(i > 0){
            layers[i]->setPrevLayer(layers[i - 1]);
        }
        if(i < layers.size() - 1){
            layers[i]->setNextLayer(layers[i + 1]);
        }
    }
}

void cn::Network::appendRecurrentLayer() {
    int id = this->layers.size();
    std::unique_ptr<RecurrentLayer> r = std::make_unique<RecurrentLayer>(id, getInputSize(id));
    layers.push_back(r.get());
    allocated.push_back(std::move(r));
}

bool cn::Network::isCudaAccelerate() const {
    return CUDAAccelerate;
}

int cn::Network::layersCount() const {
    return layers.size();
}
