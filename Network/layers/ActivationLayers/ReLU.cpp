//
// Created by user on 06.09.2021.
//

#include "ReLU.h"
#include "../../Network.h"

cn::Bitmap<double> cn::ReLU::run(const cn::Bitmap<double> &input) {
    Bitmap<double> result(input.size());
    for(int z = 0; z < input.d(); z ++){
        for(int y = 0; y < input.h(); y ++){
            for(int x = 0; x < input.w(); x ++){
                result.setCell(x, y, z, relu(input.getCell(x, y, z)));
            }
        }
    }
    return result;
}

double cn::ReLU::getChain(const Vector3<int> &inputPos) {
    const Bitmap<double> &input = network->getInput(__id);
    return diff(input.getCell(inputPos)) * network->getChain(__id + 1, inputPos);
}

cn::JSON cn::ReLU::jsonEncode() const {
    return Layer::jsonEncode();
}

double cn::ReLU::relu(double x) {
    return x > 0 ? x : 0;;
}

double cn::ReLU::diff(double x) {
    return x > 0 ? 1 : 0;
}

cn::ReLU::ReLU(int id, cn::Network &network) : Layer(id, network) {
    outputSize = inputSize;
}
