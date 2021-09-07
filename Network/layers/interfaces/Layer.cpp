//
// Created by jrazek on 27.07.2021.
//

#include "Layer.h"
#include "../../Network.h"
#include "../../layers/ConvolutionLayer.h"
#include "../../layers/FFLayer.h"
#include "../../layers/BatchNormalizationLayer.h"

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

    using result_type = std::unique_ptr<Layer>;
    using sv = std::string_view;
    using callback_type = std::function<result_type(const sv&)>;
    auto deserializerCallbacks = std::map<std::string_view, callback_type>();

    deserializerCallbacks["cl"] = [&](const cn::JSON &json) {
        std::cout << "creating convolutional_layer\nUsing: ";
        return std::make_unique<ConvolutionLayer>(network, json);
    };

    deserializerCallbacks["ffl"] = [&](const cn::JSON &json) {
        std::cout << "creating ff_layer\nUsing: ";
        return std::make_unique<FFLayer>(network, json);
    };

    deserializerCallbacks["bnl"] = [&](const cn::JSON &json) {
        std::cout << "creating batch normalization layer\nUsing: ";
        std::cout << '\n';
        return std::make_unique<BatchNormalizationLayer>(network, json);
    };

    deserializerCallbacks["fl"] = [&](const cn::JSON &json) {
        std::cout << "creating batch flattening layer\nUsing: ";
        std::cout << '\n';
        return std::make_unique<FlatteningLayer>(network, json);
    };

    return std::unique_ptr<Layer>();
}

