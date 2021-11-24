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
#include "layers/ActivationLayers/Softmax/Softmax.h"
namespace cn {
    void cn::Network::feed(Tensor<double> bitmap) {
#ifndef NNL_WITH_CUDA
        if(CUDAAccelerate){
            throw std::logic_error("CUDA IS NOT AVAILABLE! set -DCOMPILE_WITH_CUDA=ON during compilation.");
        }
#endif
        if (bitmap.size() != inputSize) {
            throw std::logic_error("invalid input size!");
        }
        if (layers.empty())
            throw std::logic_error("network must have at least one layer in order to feed it!");

        const Tensor<double> *_input = &bitmap;
        for (u_int i = 0; i < layers.size(); i++) {

            auto layer = layers[i].get();
            if (!CUDAAccelerate)
                layer->CPURun(*_input);
            else
                layer->CUDARun(*_input);
            _input = &getOutput(i, layer->getTime());
        }

        for (auto &l : layers)
            l->incTime();
    }

    void Network::feed(const Tensor<byte> &bitmap) {
        feed(Utils::normalize(bitmap));
    }

    void Network::initRandom() {
        for (auto &l : learnableLayers) {
            l->randomInit(randomEngine);
        }
    }

    void Network::appendConvolutionLayer(Vector2<int> kernelSize, int kernelsCount, Vector2<int> stride, Vector2<int> padding) {

        int id = this->layers.size();
        std::unique_ptr<ConvolutionLayer> c = std::make_unique<ConvolutionLayer>(getInputSize(id), kernelSize, kernelsCount, stride, padding);
        learnableLayers.push_back(c.get());
        layers.push_back(std::move(c));
    }

    void Network::appendFFLayer(int neuronsCount) {
        int id = this->layers.size();
        std::unique_ptr<FFLayer> f = std::make_unique<FFLayer>(getInputSize(id), neuronsCount);
        learnableLayers.push_back(f.get());
        layers.push_back(std::move(f));
    }

    void Network::appendFlatteningLayer() {
        int id = this->layers.size();
        std::unique_ptr<FlatteningLayer> f = std::make_unique<FlatteningLayer>(getInputSize(id));
        layers.push_back(std::move(f));
    }

    void Network::appendBatchNormalizationLayer() {
        int id = this->layers.size();
        std::unique_ptr<BatchNormalizationLayer> b = std::make_unique<BatchNormalizationLayer>(getInputSize(id));
        layers.push_back(std::move(b));
    }

    void Network::appendMaxPoolingLayer(Vector2<int> kernelSize) {
        int id = this->layers.size();
        std::unique_ptr<MaxPoolingLayer> m = std::make_unique<MaxPoolingLayer>(getInputSize(id), kernelSize);
        layers.push_back(std::move(m));
    }

    void Network::appendReLULayer() {
        int id = this->layers.size();
        std::unique_ptr<ReLU> r = std::make_unique<ReLU>(getInputSize(id));
        layers.push_back(std::move(r));
    }

    void Network::appendSigmoidLayer() {
        int id = this->layers.size();
        std::unique_ptr<Sigmoid> s = std::make_unique<Sigmoid>(getInputSize(id));
        layers.push_back(std::move(s));
    }

    void Network::appendSoftmaxLayer() {
        std::unique_ptr<Softmax> s = std::make_unique<Softmax>(layers.back()->getOutputSize());
        layers.push_back(std::move(s));
    }

    void Network::ready() {
        u_int id = layers.size();
        if (!outputLayer) {
            std::unique_ptr<OutputLayer> _outputLayer = std::make_unique<OutputLayer>(getInputSize(id));
            outputLayer = _outputLayer.get();
            layers.push_back(std::move(_outputLayer));
        }
        linkLayers(layers);
        resetState();
    }

    void Network::resetState() {
        for (auto &l : layers) {
            l->resetState();
        }
    }

    Vector3<int> Network::getOutputSize(int layerID) const {
        return layers[layerID]->getOutputSize();
    }

    Vector3<int> Network::getInputSize(int layerID) const {
        if (layerID == 0) {
            return inputSize;
        }
        return getOutputSize(layerID - 1);
    }

    const Tensor<double> &Network::getNetworkOutput(int time) const {
        return layers.back()->getOutput(time);
    }

    OutputLayer &cn::Network::getOutputLayer() {
        return *outputLayer;
    }

    const Tensor<double> &Network::getInput(int layerID, int time) const {
        if (layerID == 0)
            return inputLayer->getInput(time);

        return getOutput(layerID - 1, time);
    }

    const Tensor<double> &Network::getOutput(int layerID, int time) const {
        return layers[layerID]->getOutput(time);
    }

    JSON cn::Network::jsonEncode() const {
        JSON jsonObject;
        jsonObject["seed"] = seed;
        jsonObject["input_size"] = inputSize.jsonEncode();
        jsonObject["layers"] = std::vector<JSON>();
        for (u_int i = 0; i < layers.size(); i++) {
            jsonObject["layers"].push_back(layers[i]->jsonEncode());
        }
        return jsonObject;
    }


    Network::Network(Vector3<int> _inputSize, int _seed, bool _CUDAAccelerate) :
            seed(_seed),
            inputSize(_inputSize),
            randomEngine(_seed) {
        std::unique_ptr<InputLayer> _inputLayer = std::make_unique<InputLayer>(inputSize);
        inputLayer = _inputLayer.get();
        CUDAAccelerate = _CUDAAccelerate;
        layers.push_back(std::move(_inputLayer));
    }

    Network::Network(int w, int h, int d, int _seed, bool _CUDAAccelerate) :
            Network(Vector3<int>(w, h, d), _seed, _CUDAAccelerate) {}

    Network::Network(const JSON &json) : seed(json.at("seed")), inputSize(json.at("input_size")) {
        JSON _layers = json.at("layers");
        for (auto &l : _layers) {
            std::unique_ptr<Layer> layer = Layer::fromJSON(l);
            if (l.contains("learnable") && l.at("learnable")) {
                learnableLayers.push_back(dynamic_cast<Learnable *>(layer.get()));
            }
            if (l.at("type") == "il") {
                inputLayer = dynamic_cast<InputLayer *>(layer.get());
            }
            std::string str = l.at("type");
            if (l.at("type") == "ol") {
                outputLayer = dynamic_cast<OutputLayer *>(layer.get());
            }
            layer->resetState();
            layers.push_back(std::move(layer));
        }
        linkLayers(layers);
    }

    Network::Network(Network &&network) :
            seed(network.seed),
            inputSize(network.inputSize),
            randomEngine(std::move(network.randomEngine)),
            layers(std::move(network.layers)),
            learnableLayers(std::move(network.learnableLayers)),
            outputLayer(network.outputLayer) {}

    Network &Network::operator=(Network &&network) {
        seed = network.seed;
        inputSize = network.inputSize;
        randomEngine = std::move(network.randomEngine);
        layers = std::move(network.layers);
        learnableLayers = std::move(network.learnableLayers);
        inputLayer = network.inputLayer;
        outputLayer = network.outputLayer;

        return *this;
    }

    void Network::linkLayers(std::vector<std::unique_ptr<Layer>> &layers) {
        for (u_int i = 0; i < layers.size(); i++) {
            if (i > 0) {
                layers[i]->setPrevLayer(layers[i - 1].get());
            }
            if (i < layers.size() - 1) {
                layers[i]->setNextLayer(layers[i + 1].get());
            }
        }
    }

    bool Network::isCudaAccelerate() const {
        return CUDAAccelerate;
    }

    int Network::layersCount() const {
        return layers.size();
    }

    void Network::appendRecurrentLayer(std::unique_ptr<RecurrentLayer> &&recurrentLayer) {
        if (recurrentLayer->getInputSize() != layers.back()->getOutputSize()) {
            throw std::logic_error("this recurrent layers has incorrect input size!");
        }
        recurrentLayer->ready();
        learnableLayers.push_back(recurrentLayer.get());
        layers.push_back(std::move(recurrentLayer));
    }

    std::unique_ptr<RecurrentLayer> Network::createRecurrentLayer() {
        return std::make_unique<RecurrentLayer>(layers.back()->getOutputSize());
    }
}