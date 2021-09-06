//
// Created by user on 06.09.2021.
//

#include "Sigmoid.h"
#include "../../Network.h"

cn::Bitmap<double> cn::Sigmoid::run(const cn::Bitmap<double> &input) {
    Bitmap<double> result(input.size());
    for(int z = 0; z < input.d(); z ++){
        for(int y = 0; y < input.h(); y ++){
            for(int x = 0; x < input.w(); x ++){
                result.setCell(x, y, z, sigmoid(input.getCell(x, y, z)));
            }
        }
    }
    return result;
}

double cn::Sigmoid::getChain(const Vector3<int> &inputPos) {
    const Bitmap<double> &input = network->getInput(__id);
    return diff(input.getCell(inputPos)) * network->getChain(__id + 1, inputPos);
}

cn::JSON cn::Sigmoid::jsonEncode() const {
    JSON structure;
    structure["id"] = __id;
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

cn::Sigmoid::Sigmoid(int id, cn::Network &network) : Layer(id, network) {
    outputSize = inputSize;
}
