//
// Created by user on 06.09.2021.
//

#include "ReLU.h"
#include "../../../Network.h"

void cn::ReLU::CPURun(const cn::Tensor<double> &input) {
    Tensor<double> result(input.size());
    for(int z = 0; z < input.d(); z ++){
        for(int y = 0; y < input.h(); y ++){
            for(int x = 0; x < input.w(); x ++){
                result.setCell(x, y, z, relu(input.getCell(x, y, z)));
            }
        }
    }
    output.push_back(Tensor<double>(std::move(result)));
}

double cn::ReLU::getChain(const Vector4<int> &inputPos) {
    const Tensor<double> &input = getInput(inputPos.t);
    return diff(input.getCell({inputPos.x, inputPos.y, inputPos.z})) * nextLayer->getChain(inputPos);
}

cn::JSON cn::ReLU::jsonEncode() const {
    JSON structure;
    structure["id"] = __id;
    structure["input_size"] = inputSize.jsonEncode();
    structure["type"] = "relu";
    return structure;
}

double cn::ReLU::relu(double x) {
    return x > 0 ? x : 0;;
}

double cn::ReLU::diff(double x) {
    return x > 0 ? 1 : 0;
}

cn::ReLU::ReLU(int id, Vector3<int> _inputSize) : Layer(id, _inputSize) {
    outputSize = inputSize;
}

cn::ReLU::ReLU(const JSON &json) :
        cn::ReLU(json.at("id"), json.at("input_size")) {}

std::unique_ptr<cn::Layer> cn::ReLU::getCopyAsUniquePtr() const {
    return std::make_unique<ReLU>(*this);
}
