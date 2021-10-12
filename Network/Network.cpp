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

        auto layer = layers[i].get();
        if(!CUDAAccelerate)
            layer->CPURun(*_input);
        else
            layer->CUDARun(*_input);
        _input = &getOutput(i, layer->getTime());
    }

    for(auto &l : layers)
        l->incTime();
}

void cn::Network::feed(const cn::Tensor<cn::byte> &bitmap) {
    feed(Utils::normalize(bitmap));
}

void cn::Network::initRandom() {
    for(auto &l : learnableLayers){
        l->randomInit(randomEngine);
    }
}

void cn::Network::appendConvolutionLayer(Vector2<int> kernelSize, int kernelsCount, Vector2<int> stride,
                                         Vector2<int> padding) {

    int id = this->layers.size();
    std::unique_ptr<ConvolutionLayer> c = std::make_unique<ConvolutionLayer>(getInputSize(id), kernelSize, kernelsCount, stride, padding);
    learnableLayers.push_back(c.get());
    layers.push_back(std::move(c));
}

void cn::Network::appendFFLayer(int neuronsCount) {
    int id = this->layers.size();
    std::unique_ptr<FFLayer> f = std::make_unique<FFLayer>(getInputSize(id), neuronsCount);
    learnableLayers.push_back(f.get());
    layers.push_back(std::move(f));
}

void cn::Network::appendFlatteningLayer() {
    int id = this->layers.size();
    std::unique_ptr<FlatteningLayer> f = std::make_unique<FlatteningLayer>(getInputSize(id));
    layers.push_back(std::move(f));
}

void cn::Network::appendBatchNormalizationLayer() {
    int id = this->layers.size();
    std::unique_ptr<BatchNormalizationLayer> b = std::make_unique<BatchNormalizationLayer>(getInputSize(id));
    layers.push_back(std::move(b));
}

void cn::Network::appendMaxPoolingLayer(Vector2<int> kernelSize) {
    int id = this->layers.size();
    std::unique_ptr<MaxPoolingLayer> m = std::make_unique<MaxPoolingLayer>(getInputSize(id), kernelSize);
    layers.push_back(std::move(m));
}

void cn::Network::appendReLULayer() {
    int id = this->layers.size();
    std::unique_ptr<ReLU> r = std::make_unique<ReLU>(getInputSize(id));
    layers.push_back(std::move(r));
}

void cn::Network::appendSigmoidLayer() {
    int id = this->layers.size();
    std::unique_ptr<Sigmoid> s = std::make_unique<Sigmoid>(getInputSize(id));
    layers.push_back(std::move(s));
}

void cn::Network::ready() {
    u_int id = layers.size();
    if(!outputLayer) {
        std::unique_ptr<OutputLayer> _outputLayer = std::make_unique<OutputLayer>(getInputSize(id));
        outputLayer = _outputLayer.get();
        layers.push_back(std::move(_outputLayer));
    }
    linkLayers(layers);
    resetState();
}

void cn::Network::resetState() {
    for(auto &l : layers){
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
    std::unique_ptr<InputLayer> _inputLayer = std::make_unique<InputLayer>(inputSize);
    inputLayer = _inputLayer.get();
    CUDAAccelerate = _CUDAAccelerate;
    layers.push_back(std::move(_inputLayer));
}

cn::Network::Network(int w, int h, int d, int _seed, bool _CUDAAccelerate):
Network(cn::Vector3<int>(w, h, d), _seed, _CUDAAccelerate)
{}

cn::Network::Network(const cn::JSON &json): seed(json.at("seed")), inputSize(json.at("input_size")) {
    JSON _layers = json.at("layers");
    for (auto &l : _layers) {
        std::unique_ptr<Layer> layer = Layer::fromJSON(l);
        if (l.contains("learnable") && l.at("learnable")) {
            learnableLayers.push_back(dynamic_cast<Learnable *>(layer.get()));
        }
        if(l.at("type") == "il"){
            inputLayer = dynamic_cast<InputLayer *>(layer.get());
        }
        std::string str = l.at("type");
        if(l.at("type") == "ol"){
            outputLayer = dynamic_cast<OutputLayer *>(layer.get());
        }
        layer->resetState();
        layers.push_back(std::move(layer));
    }
    linkLayers(layers);
}

cn::Network::Network(cn::Network &&network):
seed(network.seed),
inputSize(network.inputSize),
randomEngine(std::move(network.randomEngine)),
layers(std::move(network.layers)),
learnableLayers(std::move(network.learnableLayers)),
outputLayer(network.outputLayer)
{}

cn::Network &cn::Network::operator=(cn::Network &&network) {
    seed = network.seed;
    inputSize = network.inputSize;
    randomEngine = std::move(network.randomEngine);
    layers = std::move(network.layers);
    learnableLayers = std::move(network.learnableLayers);
    inputLayer = network.inputLayer;
    outputLayer = network.outputLayer;

    return *this;
}

void cn::Network::linkLayers(std::vector<std::unique_ptr<Layer>> &layers) {
    for(u_int i = 0; i < layers.size(); i ++){
        if(i > 0){
            layers[i]->setPrevLayer(layers[i - 1].get());
        }
        if(i < layers.size() - 1){
            layers[i]->setNextLayer(layers[i + 1].get());
        }
    }
}

bool cn::Network::isCudaAccelerate() const {
    return CUDAAccelerate;
}

int cn::Network::layersCount() const {
    return layers.size();
}

void cn::Network::appendRecurrentLayer(std::unique_ptr<RecurrentLayer> &&recurrentLayer) {
    if(recurrentLayer->getInputSize() != layers.back()->getOutputSize()){
        throw std::logic_error("this recurrent layers has incorrect input size!");
    }
    recurrentLayer->ready();
    learnableLayers.push_back(recurrentLayer.get());
    layers.push_back(std::move(recurrentLayer));
}

std::unique_ptr<cn::RecurrentLayer> cn::Network::createRecurrentLayer() {
    return std::make_unique<RecurrentLayer>(layers.back()->getOutputSize());
}
