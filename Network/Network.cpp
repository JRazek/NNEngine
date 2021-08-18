//
// Created by jrazek on 27.07.2021.
//

#include <stdexcept>
#include "Network.h"
#include "layers/ConvolutionLayer.h"

void cn::Network::appendLayer(cn::Layer * layer) {
    //todo validation!
    this->layers.push_back(layer);
}

void cn::Network::feed(const byte *input) {
    cn::Bitmap<byte> bitmap(inputDataWidth, inputDataHeight, inputDataHeight, input, 0);
    if(this->layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");
    feed(cn::Utils::normalize(bitmap));
}

cn::Network::~Network() {
    for(auto l : this->layers){
        delete l;
    }
}

void cn::Network::appendConvolutionLayer(int kernelX, int kernelY, int kernelZ, int kernelsCount, const DifferentiableFunction &function, int paddingX,
                                         int paddingY, int strideX, int strideY) {


    this->layers.push_back(new ConvolutionLayer(this->layers.size(), this, kernelX, kernelY, kernelZ, kernelsCount,
                                                function, paddingX, paddingY, strideX, strideY));
}

const std::vector<cn::Layer *> *cn::Network::getLayers() {
    return &this->layers;
}

cn::Network::Network(int w, int h, int d): inputDataWidth(w), inputDataHeight(h), inputDataDepth(d) {

}

void cn::Network::feed(const cn::Bitmap<float> &bitmap) {
    cn::Bitmap<float> resized = cn::Utils::resize<float>(bitmap, inputDataWidth, inputDataHeight);

    if(this->layers.empty())
        throw std::logic_error("network must have at least one layer in order to feed it!");
    const Bitmap<float> * input = &resized;
    for(int i = 0; i < layers.size(); i ++){
        auto layer = layers[i];
        layer->run(*input);
        input = layer->output;
    }
}

void cn::Network::feed(const cn::Bitmap<cn::byte> &bitmap) {
    feed(cn::Utils::normalize(bitmap));
}
