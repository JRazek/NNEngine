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
#include "layers/InputLayer.h"
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
    for(u_int i = 0; i < layers.size(); i ++){
        auto layer = layers[i];
        layer->run(*_input);
        _input = &getOutput(i).value();
    }
}

void cn::Network::feed(const cn::Bitmap<cn::byte> &bitmap) {
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

void cn::Network::appendReluLayer() {
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
    if(!outputLayer.has_value()) {
        outputLayer.emplace(id, getInputSize(id));
        layers.push_back(&outputLayer.value());
    }
    for(u_int i = 0; i < layers.size(); i ++){
        if(i > 0){
            layers[i]->setPrevLayer(layers[i - 1]);
        }
        if(i < layers.size() - 1){
            layers[i]->setNextLayer(layers[i + 1]);
        }
    }
    int tret = 04;
}

const std::vector<cn::Learnable *> &cn::Network::getLearnables() const{
    return learnableLayers;
}

void cn::Network::resetMemoization() {
    for(auto l : layers){
        l->resetMemoization();
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

const std::optional<cn::Bitmap<double>> &cn::Network::getNetworkOutput() const {
    return layers.back()->getOutput();
}

cn::OutputLayer &cn::Network::getOutputLayer() {
    return outputLayer.value();
}

const std::optional<cn::Bitmap<double>> &cn::Network::getInput(int layerID) const{
    if(layerID == 0)
        return input;

    return getOutput(layerID -1);
}

const std::optional<cn::Bitmap<double>> &cn::Network::getOutput(int layerID) const {
    return layers[layerID]->getOutput();
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
        randomEngine(_seed){
    std::unique_ptr<Layer> inputLayer = std::make_unique<InputLayer>(0, inputSize);
    layers.push_back(inputLayer.get());
    allocated.push_back(std::move(inputLayer));
}

cn::Network::Network(int w, int h, int d, int _seed):
Network(cn::Vector3<int>(w, h, d), _seed)
{}

cn::Network::Network(const cn::JSON &json): seed(json.at("seed")), inputSize(json.at("input_size")) {
    try {
        JSON _layers = json.at("layers");
        for (auto l : _layers) {
            if (l.at("type") == "ol") {
                outputLayer.emplace(OutputLayer(l));
                layers.push_back(&outputLayer.value());
            } else {
                allocated.push_back(Layer::fromJSON(l));
                layers.push_back(allocated.back().get());
            }
            if (l.contains("learnable") && l.at("learnable")) {
                learnableLayers.push_back(dynamic_cast<Learnable *>(layers.back()));
            }
        }
        ready();
    }catch(std::exception &e){
        std::cout<<e.what();
    }
}

cn::Network::Network(cn::Network &&network) {
    *this = std::move(network);
}

cn::Network &cn::Network::operator=(cn::Network &&network) {
    layers.clear();
    allocated.clear();
    learnableLayers.clear();

    seed = network.seed;
    inputSize = network.inputSize;
    randomEngine = std::move(network.randomEngine);
    allocated = std::move(network.allocated);
    learnableLayers = std::move(network.learnableLayers);
    outputLayer = network.outputLayer;
    for(auto &a : allocated){
        layers.push_back(a.get());
    }
    layers.push_back(&outputLayer.value());

    input = std::move(network.input);
    ready();

    return *this;
}
