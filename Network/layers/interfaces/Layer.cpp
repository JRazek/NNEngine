//
// Created by jrazek on 27.07.2021.
//

#include "Layer.h"
#include "../../Network.h"
#include "../../layers/ConvolutionLayer.h"
#include "../../layers/FFLayer.h"
#include "../../layers/BatchNormalizationLayer.h"
#include "../../layers/MaxPoolingLayer.h"
#include "../../layers/ActivationLayers/ReLU.h"
#include "../../layers/ActivationLayers/Sigmoid.h"

cn::Layer::Layer(int _id, Network &_network): network(&_network), __id(_id){
    inputSize = network->getInputSize(_id);
    memoizationStates.emplace(Bitmap<bool>(inputSize));
    memoizationTable.emplace(Bitmap<double>(inputSize));
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

std::unique_ptr<cn::Layer> cn::Layer::fromJSON(Network &network, const cn::JSON &json) {

    using callback_type = std::function<std::unique_ptr<Layer>(const cn::JSON &json)>;

    std::unordered_map<std::string_view, callback_type> deserializerCallbacks;

    deserializerCallbacks["cl"] = [&](const cn::JSON &json) {
        std::cout << "creating convolutional_layer \n";
        return std::make_unique<ConvolutionLayer>(network, json);
    };

    deserializerCallbacks["ffl"] = [&](const cn::JSON &json) {
        std::cout << "creating ff_layer \n";
        return std::make_unique<FFLayer>(network, json);
    };

    deserializerCallbacks["bnl"] = [&](const cn::JSON &json) {
        std::cout << "creating batch normalization layer \n";
        return std::make_unique<BatchNormalizationLayer>(network, json);
    };

    deserializerCallbacks["fl"] = [&](const cn::JSON &json) {
        std::cout << "creating flattening layer \n";
        return std::make_unique<FlatteningLayer>(network, json);
    };

    deserializerCallbacks["mpl"] = [&](const cn::JSON &json) {
        std::cout << "creating max pooling layer \n";
        return std::make_unique<MaxPoolingLayer>(network, json);
    };

    deserializerCallbacks["relu"] = [&](const cn::JSON &json) {
        std::cout << "creating relu layer \n";
        return std::make_unique<ReLU>(network, json);
    };

    deserializerCallbacks["sig"] = [&](const cn::JSON &json) {
        std::cout << "creating sigmoid layer \n";
        return std::make_unique<Sigmoid>(network, json);
    };

    deserializerCallbacks["ol"] = [&](const cn::JSON &json) {
        std::cout << "creating output layer \n";
        return std::make_unique<OutputLayer>(network, json);
    };

    const auto deserialize = [&](std::string_view type, const cn::JSON &json) {
        return deserializerCallbacks[type](json);
    };

    std::string str = json["type"];

    return deserialize(str, json);
}

