//
// Created by user on 06.09.2021.
//

#include "Sigmoid.h"
#include "../../../Network.h"

void cn::Sigmoid::CPURun(const cn::Bitmap<double> &input) {
    Bitmap<double> result(input.size());
    for(int z = 0; z < input.d(); z ++){
        for(int y = 0; y < input.h(); y ++){
            for(int x = 0; x < input.w(); x ++){
                result.setCell(x, y, z, sigmoid(input.getCell(x, y, z)));
            }
        }
    }
    output.emplace(std::move(result));
}

double cn::Sigmoid::getChain(const Vector3<int> &inputPos) {
    const Bitmap<double> &input = getInput().value();
    return diff(input.getCell(inputPos)) * nextLayer->getChain(inputPos);
}

cn::JSON cn::Sigmoid::jsonEncode() const {
    JSON structure;
    structure["id"] = __id;
    structure["input_size"] = inputSize.jsonEncode();
    structure["type"] = "sig";
    return structure;
}

double cn::Sigmoid::sigmoid(double x) {
    return 1.f/(1.f + std::pow(Sigmoid::e, -x));
}

double cn::Sigmoid::diff(double x) {
    double sig = sigmoid(x);
    return sig * (1.f - sig);
}

cn::Sigmoid::Sigmoid(int id, Vector3<int> _inputSize) :
Layer(id, _inputSize) {
    outputSize = inputSize;
}

cn::Sigmoid::Sigmoid(const JSON &json) :
Sigmoid(json.at("id"), json.at("input_size")) {}

std::unique_ptr<cn::Layer> cn::Sigmoid::getCopyAsUniquePtr() const {
    return std::make_unique<Sigmoid>(*this);
}
