//
// Created by jrazek on 27.07.2021.
//

#include "Layer.h"
#include "../../Network.h"
#include "../ConvolutionLayer/ConvolutionLayer.h"
#include "../FFLayer/FFLayer.h"
#include "../BatchNormalizationLayer/BatchNormalizationLayer.h"
#include "../MaxPoolingLayer/MaxPoolingLayer.h"
#include "../ActivationLayers/ReLU/ReLU.h"
#include "../ActivationLayers/Sigmoid/Sigmoid.h"
#include "../InputLayer/InputLayer.h"

cn::Layer::Layer(int _id, Vector3<int> _inputSize) :
inputSize(_inputSize), __id(_id){
    memoizationStates = std::make_unique<Bitmap<bool>>(Bitmap<bool>(inputSize));
    memoizationTable = std::make_unique<Bitmap<double>>(Bitmap<double>(inputSize));
    resetMemoization();
}

[[maybe_unused]] int cn::Layer::id() const {
    return __id;
}

void cn::Layer::setMemo(const Vector3<int> &pos, double val) {
    memoizationStates->setCell(pos, true);
    memoizationTable->setCell(pos, val);
}

void cn::Layer::resetMemoization() {
    std::fill(memoizationStates->data(), memoizationStates->data() + memoizationStates->size().multiplyContent(), false);
}

double cn::Layer::getMemo(const Vector3<int> &pos) const {
    return memoizationTable->getCell(pos);
}

bool cn::Layer::getMemoState(const Vector3<int> &pos) const {
    return memoizationStates->getCell(pos);
}

cn::Vector3<int> cn::Layer::getOutputSize() const {
    return outputSize;
}

cn::JSON cn::Layer::jsonEncode() const{
    return JSON();
}

std::unique_ptr<cn::Layer> cn::Layer::fromJSON(const cn::JSON &json) {

    using callback_type = std::function<std::unique_ptr<Layer>(const cn::JSON &json)>;

    std::unordered_map<std::string_view, callback_type> deserializerCallbacks;

    deserializerCallbacks["cl"] = [&](const cn::JSON &json) {
        std::cout << "creating convolutional_layer \n";
        return std::make_unique<ConvolutionLayer>(json);
    };

    deserializerCallbacks["ffl"] = [&](const cn::JSON &json) {
        std::cout << "creating ff_layer \n";
        return std::make_unique<FFLayer>(json);
    };

    deserializerCallbacks["bnl"] = [&](const cn::JSON &json) {
        std::cout << "creating batch normalization layer \n";
        return std::make_unique<BatchNormalizationLayer>(json);
    };

    deserializerCallbacks["fl"] = [&](const cn::JSON &json) {
        std::cout << "creating flattening layer \n";
        return std::make_unique<FlatteningLayer>(json);
    };

    deserializerCallbacks["mpl"] = [&](const cn::JSON &json) {
        std::cout << "creating max pooling layer \n";
        return std::make_unique<MaxPoolingLayer>(json);
    };

    deserializerCallbacks["relu"] = [&](const cn::JSON &json) {
        std::cout << "creating relu layer \n";
        return std::make_unique<ReLU>(json);
    };

    deserializerCallbacks["sig"] = [&](const cn::JSON &json) {
        std::cout << "creating sigmoid layer \n";
        return std::make_unique<Sigmoid>(json);
    };

    deserializerCallbacks["ol"] = [&](const cn::JSON &json) {
        std::cout << "creating output layer \n";
        return std::make_unique<OutputLayer>(json);
    };

    deserializerCallbacks["il"] = [&](const cn::JSON &json) {
        std::cout << "creating input layer \n";
        return std::make_unique<InputLayer>(json);
    };

    const auto deserialize = [&](std::string_view type, const cn::JSON &json) {
        return deserializerCallbacks[type](json);
    };

    std::string str = json["type"];

    return deserialize(str, json);
}

void cn::Layer::setPrevLayer(cn::Layer *_prevLayer) {
    prevLayer = _prevLayer;
}

void cn::Layer::setNextLayer(cn::Layer *_nextLayer) {
    nextLayer = _nextLayer;
}

const std::unique_ptr<cn::Bitmap<double>> &cn::Layer::getOutput() const {
    return output;
}

const std::unique_ptr<cn::Bitmap<double>> &cn::Layer::getInput() const {
    return prevLayer->getOutput();
}

void cn::Layer::CUDARun(const cn::Bitmap<double> &_input) {
    CPURun(_input);
    ///placeholder
}

cn::Layer::Layer(const cn::Layer &layer) :
inputSize(layer.inputSize), __id(layer.id()){
    memoizationStates = std::make_unique<Bitmap<bool>>(*layer.memoizationStates.get());
    memoizationTable = std::make_unique<Bitmap<double>>(*layer.memoizationTable.get());
}

cn::Layer::Layer(cn::Layer &&layer) :
inputSize(layer.inputSize), __id(layer.id()){
    memoizationStates = std::move(layer.memoizationStates);
    memoizationTable = std::move(layer.memoizationTable);
}

void cn::Layer::CUDAAutoGrad() {
    throw std::logic_error("this should be overridden!");
}
