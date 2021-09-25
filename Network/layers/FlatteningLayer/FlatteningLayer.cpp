//
// Created by jrazek on 19.08.2021.
//

#include "FlatteningLayer.h"
#include "../../Network.h"

cn::FlatteningLayer::FlatteningLayer(int _id, Vector3<int> _inputSize) : Layer(_id, _inputSize) {
    int size = inputSize.multiplyContent();
    outputSize = Vector3<int>(size, 1, 1);
}

void cn::FlatteningLayer::CPURun(const cn::Tensor<double> &input) {
    if(input.size().multiplyContent() != inputSize.multiplyContent())
        throw std::logic_error("invalid input input for flattening layer!");
    Tensor<double> result ({inputSize.multiplyContent(), 1, 1}, input.dataConst());
    output = std::make_unique<Tensor<double>>(std::move(result));
}

double cn::FlatteningLayer::getChain(const Vector3<int> &inputPos) {
    if(getMemoState(inputPos)){
        return getMemo(inputPos);
    }
    int outputIndex = prevLayer->getOutput()->getDataIndex(inputPos);
    double res = nextLayer->getChain({outputIndex, 0, 0});
    setMemo(inputPos, res);
    return res;
}

cn::JSON cn::FlatteningLayer::jsonEncode() const {
    JSON structure;
    structure["id"] = __id;
    structure["input_size"] = inputSize.jsonEncode();
    structure["type"] = "fl";
    return structure;
}

cn::FlatteningLayer::FlatteningLayer(const JSON &json) :
FlatteningLayer(json.at("id"), json.at("input_size")) {}

std::unique_ptr<cn::Layer> cn::FlatteningLayer::getCopyAsUniquePtr() const {
    return std::make_unique<FlatteningLayer>(*this);
}
