//
// Created by user on 20.08.2021.
//

#include "MaxPoolingLayer.h"
#include "../Network.h"

cn::MaxPoolingLayer::MaxPoolingLayer(int _id, Network &_network, int _kernelSizeX, int _kernelSizeY) : MaxPoolingLayer(_id, _network, {_kernelSizeX, _kernelSizeY}){
}

cn::Bitmap<double> cn::MaxPoolingLayer::run(const cn::Bitmap<double> &input) {
    if(input.size() != inputSize){
        throw std::logic_error("invalid output size in max pool!");
    }

    std::fill(mapping->data(), mapping->data() + mapping->size().multiplyContent(), Vector2<int>(-1, -1));

    Bitmap<double> res(outputSize);

    for(int c = 0; c < input.d(); c++){
        for(int y = 0; y < input.h() - kernelSize.y + 1; y += kernelSize.y){
            for(int x = 0; x < input.w() - kernelSize.x + 1; x += kernelSize.x){
                double max = 0;
                Vector2<int> bestPoint;
                for(int kY = 0; kY < kernelSize.y; kY++){
                    for(int kX = 0; kX < kernelSize.x; kX++){
                        max = std::max(input.getCell(x + kX, y + kY, c), max);
                        bestPoint = Vector2<int>(x + kX, y + kY);
                    }
                }
                res.setCell(x / kernelSize.x, y / kernelSize.y, c, max);
                mapping->setCell(bestPoint.x, bestPoint.x, c, {x / kernelSize.x, y / kernelSize.y});
            }
        }
    }
    return res;
}

double cn::MaxPoolingLayer::getChain(const Vector3<int> &inputPos) {
    if(getMemoState(inputPos)){
        return getMemo(inputPos);
    }
    Vector2<int> mapped = mapping->getCell(inputPos);
    double res;
    if(mapped == Vector2<int>(-1, -1))
        res = 0;
    else
        res = network->getChain(__id + 1, {mapped.x, mapped.y, inputPos.z});
    setMemo(inputPos, res);
    return res;
}

cn::JSON cn::MaxPoolingLayer::jsonEncode() const {
    JSON structure;
    structure["id"] = __id;
    structure["type"] = "mpl";
    structure["kernel_size"] = kernelSize.jsonEncode();
    return structure;
}

cn::MaxPoolingLayer::MaxPoolingLayer(cn::Network &_network, const cn::JSON &json):
MaxPoolingLayer(json.at("id"), _network, json.at("kernel_size")){}

cn::MaxPoolingLayer::MaxPoolingLayer(int _id, cn::Network &_network, cn::Vector2<int> _kernelSize):
Layer(_id, _network),
kernelSize(_kernelSize){
    outputSize = Vector3<int>(Utils::afterMaxPoolSize(kernelSize.x, inputSize.x), Utils::afterMaxPoolSize(kernelSize.y, inputSize.y), inputSize.z);
    mapping.emplace(Bitmap<Vector2<int>>(inputSize));
}

std::unique_ptr<cn::Layer> cn::MaxPoolingLayer::getCopyAsUniquePtr() const {
    return std::make_unique<MaxPoolingLayer>(*this);
}
