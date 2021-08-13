//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "Network.h"
#include "layers/ConvolutionLayer.h"
#include "../Utils/Utils.h"

void cn::Network::appendLayer(cn::Layer * layer) {
    //todo validation!
    this->layers.push_back(layer);
}

void cn::Network::feed(const byte *input) {
    cn::Bitmap<byte> bitmap(this->inputDataWidth, this->inputDataHeight, this->inputDataHeight, input, 0);
    //normalize image
    cn::Bitmap<float> normalized = cn::Utils::normalize(bitmap);
    if(this->layers.empty())
        throw std::logic_error("network must have at least layer in order to feed it!");

    //todo override = operator
//    normalized = cn::Utils::resize<float>(normalized, 1, 1);
    cn::Bitmap<float> resized = cn::Utils::resize<float>(normalized, inputDataWidth, inputDataHeight);

    this->layers.front()->run(resized);
}

cn::Network::~Network() {
    for(auto l : this->layers){
        delete l;
    }
}

void cn::Network::appendConvolutionLayer(int kernelX, int kernelY, int kernelZ, int kernelsCount, int paddingX,
                                         int paddingY) {
    this->layers.push_back(new ConvolutionLayer(this->layers.size(), this, kernelX, kernelY, kernelZ, kernelsCount, paddingX, paddingY));
}

const std::vector<cn::Layer *> *cn::Network::getLayers() {
    return &this->layers;
}

cn::Network::Network(int w, int h, int d): inputDataWidth(w), inputDataHeight(h), inputDataDepth(d) {

}

void cn::Network::feed(const cn::Bitmap<float> &bitmap) {
    if(this->layers.empty())
        throw std::logic_error("network must have at least layer in order to feed it!");
    this->layers.front()->run(bitmap);
}
