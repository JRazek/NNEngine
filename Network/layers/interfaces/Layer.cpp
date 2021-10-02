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
    resetState();
}

[[maybe_unused]] int cn::Layer::id() const {
    return __id;
}

void cn::Layer::setMemo(const Vector4<int> &pos, double val) {
    memoizationStates[pos.t].setCell({pos.x, pos.y, pos.z}, true);
    memoizationTable[pos.t].setCell({pos.x, pos.y, pos.z}, val);
}

void cn::Layer::resetState() {
    _time = 0;
    output.clear();
    memoizationTable.clear();
    memoizationStates.clear();
}

double cn::Layer::getMemo(const Vector4<int> &pos) const {
    return memoizationTable[pos.t].getCell({pos.x, pos.y, pos.z});
}

bool cn::Layer::getMemoState(const Vector4<int> &pos) const {
    return memoizationStates[pos.t].getCell({pos.x, pos.y, pos.z});
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

const cn::Tensor<double> &cn::Layer::getOutput(int time) const {
    return output[time];
}

const cn::Tensor<double> &cn::Layer::getInput(int time) const {
    return prevLayer->getOutput(time);
}

void cn::Layer::CUDARun(const cn::Tensor<double> &_input) {
    CPURun(_input);
    ///placeholder
}

cn::Layer::Layer(const cn::Layer &layer) :
inputSize(layer.inputSize), __id(layer.id()){}

cn::Layer::Layer(cn::Layer &&layer) :
inputSize(layer.inputSize), __id(layer.id()){
    memoizationStates = std::move(layer.memoizationStates);
    memoizationTable = std::move(layer.memoizationTable);
}

void cn::Layer::CUDAAutoGrad() {
    throw std::logic_error("this should be overridden!");
}

void cn::Layer::addMemoLayer() {
    Tensor<bool> states(inputSize);
    Tensor<double> table(inputSize);

    std::fill(states.data(), states.data() + states.size().multiplyContent(), 0);
    std::fill(table.data(), table.data() + table.size().multiplyContent(), 0);

    memoizationStates.push_back(std::move(states));
    memoizationTable.push_back(std::move(table));
}

void cn::Layer::incTime() {
    ++_time;
}

int cn::Layer::getTime() const{
    return _time;
}
