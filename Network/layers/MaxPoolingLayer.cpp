//
// Created by user on 20.08.2021.
//

#include "MaxPoolingLayer.h"
#include "../Network.h"

cn::MaxPoolingLayer::MaxPoolingLayer(int _id, Network &_network, int _kernelSizeX, int _kernelSizeY) :
        Layer(_id, _network),
        kernelSizeX(_kernelSizeX),
        kernelSizeY(_kernelSizeY){
    int inputX, inputY, sizeX, sizeY, sizeZ;
    if(__id == 0){
        inputX = network->inputDataWidth;
        inputY = network->inputDataHeight;
        sizeX = Utils::afterMaxPoolSize(kernelSizeX, inputX);
        sizeY = Utils::afterMaxPoolSize(kernelSizeY, inputY);
        sizeZ = network->inputDataDepth;
    }else{
        const Bitmap<float> &prev = network->getInput(__id);
        inputX = prev.w();
        inputY = prev.h();
        sizeX = Utils::afterMaxPoolSize(kernelSizeX, inputX);
        sizeY = Utils::afterMaxPoolSize(kernelSizeY, inputY);
        sizeZ = prev.d();
    }
    output.emplace(Bitmap<float>(sizeX, sizeY, sizeZ));
    mapping.emplace(Bitmap<Vector2<int>>(inputX, inputY, sizeZ));
}

void cn::MaxPoolingLayer::run(const cn::Bitmap<float> &input) {
    _input = &input;
    std::fill(mapping->data(), mapping->data() + mapping->w() * mapping->h() * mapping->d(), Vector2<int>(-1, -1));
    Bitmap<float> res(Utils::afterMaxPoolSize(kernelSizeX, input.w()), Utils::afterMaxPoolSize(kernelSizeY, input.h()), input.d());
    for(int c = 0; c < input.d(); c++){
        for(int y = 0; y < input.h() - kernelSizeY; y += kernelSizeY){
            for(int x = 0; x < input.w() - kernelSizeX; x += kernelSizeX){
                float max = 0;
                Vector2<int> bestPoint;
                for(int kY = 0; kY < kernelSizeY; kY++){
                    for(int kX = 0; kX < kernelSizeX; kX++){
                        max = std::max(input.getCell(x + kX, y + kY, c), max);
                        bestPoint = Vector2<int>(x + kX, y + kY);
                    }
                }
                res.setCell(x / kernelSizeX, y / kernelSizeY, c, max);
                mapping->setCell(bestPoint.x, bestPoint.x, c, {x / kernelSizeX, y / kernelSizeY});
            }
        }
    }

    if(res.w() != output->w() || res.h() != output->h() || res.d() != output->d()){
        throw std::logic_error("invalid output size in max pool!");
    }
    std::copy(res.data(), res.data() + res.w() * res.h() * res.d(), output->data());
}

float cn::MaxPoolingLayer::getChain(const Vector3<int> &inputPos) {
    if(getMemoState(inputPos)){
        return getMemo(inputPos);
    }
    Vector2<int> mapped = mapping->getCell(inputPos);
    float res;
    if(mapped == Vector2<int>(-1, -1))
        res = 0;
    else
        res = network->getChain(__id + 1, {mapped.x, mapped.y, inputPos.z});
    setMemo(inputPos, res);
    return res;
}
