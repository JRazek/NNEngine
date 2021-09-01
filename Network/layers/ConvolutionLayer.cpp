//
// Created by jrazek on 27.07.2021.
//
#include "ConvolutionLayer.h"
#include "../Network.h"
#include <future>
cn::ConvolutionLayer::ConvolutionLayer(int _id, Network &_network, int _kernelSizeX, int _kernelSizeY,
                                       int _kernelsCount, const DifferentiableFunction &_activationFunction,
                                       int _paddingX, int _paddingY, int _strideX, int _strideY) :
        kernelSize(_kernelSizeX, _kernelSizeY, network->getInputSize(_id).z),
        kernelsCount(_kernelsCount),
        activationFunction(_activationFunction),
        paddingX(_paddingX),
        paddingY(_paddingY),
        strideX(_strideX),
        strideY(_strideY),
        biases(kernelsCount),
        cn::Learnable(_id, _network, _kernelsCount) {

    if(inputSize.x < kernelSize.x || inputSize.y < kernelSize.y){
        throw std::logic_error("kernel must not be larger than input!");
    }
    kernels.reserve(_kernelsCount);

    for(int i = 0; i < _kernelsCount; i ++){
        kernels.emplace_back(kernelSize);
    }

    int oX = Utils::afterConvolutionSize(kernelSize.x, inputSize.x, paddingX, strideX);
    int oY = Utils::afterConvolutionSize(kernelSize.y, inputSize.y, paddingY, strideY);
    int oZ = kernelsCount;
    outputSize = Vector3<int>(oX, oY, oZ);
    beforeActivation.emplace(outputSize);
}

cn::Bitmap<float> cn::ConvolutionLayer::run(const Bitmap<float> &input) {
    if(input.size() != network->getInputSize(__id)){
        throw std::logic_error("CLayer fed with wrong input size!");
    }
    std::vector<std::future<Bitmap<float>>> kernelThreads;
    auto getConvolved = [this](const Bitmap<float> &input, const cn::Bitmap<float> &kernel){
        return Utils::sumBitmapLayers(Utils::convolve(kernel, input, paddingX, paddingY, strideX, strideY));
    };
    for(int i = 0; i < kernelsCount; i ++){
        kernelThreads.push_back(std::async(getConvolved, input, kernels[i]));
    }
    for(int i = 0; i < kernelsCount; i ++){
        beforeActivation->setLayer(i, kernelThreads[i].get().data());
    }
    Bitmap<float> result = beforeActivation.value();
    for(auto it = beforeActivation->data(); it != beforeActivation->data() + beforeActivation->w() * beforeActivation->h() * beforeActivation->d(); ++it){
        int index = it - beforeActivation->data();
        *(result.data() + index) = activationFunction.func(*it);
    }
    return result;
}

void cn::ConvolutionLayer::randomInit() {
    for(auto &k : kernels){
        for(auto it = k.data(); it != k.data() + k.w() * k.h() * k.d(); ++it){
            *it = network->getRandom(0, 1);
        }
    }
    for(auto &b : biases){
        b = network->getRandom(0, 0);
    }
}

float cn::ConvolutionLayer::getChain(const Vector3<int> &inputPos) {
    if(getMemoState(inputPos)){
        return getMemo(inputPos);
    }
    Bitmap<float> paddedInput = Utils::addPadding(*network->getInput(__id), paddingX, paddingY);

    auto validPos = [this](const Vector2<int> &kernelPos, const Bitmap<float> &bitmap){
        return kernelPos.x >= 0 && kernelPos.y >= 0 && kernelPos.x + kernelSize.x - 1 < bitmap.w() && kernelPos.y + kernelSize.y - 1 < bitmap.h();
    };

    float result = 0;

    Vector3<int> inputPosPadded(inputPos.x - paddingX, inputPos.y - paddingY, inputPos.z);
    for(int c = 0; c < kernelsCount; c++){
        for(int y = 0; y < kernelSize.y; y++){
            for(int x = 0; x < kernelSize.x; x++){
                Vector2<int> kernelPos(inputPosPadded.x - x, inputPosPadded.y - y);
                if(validPos(kernelPos, paddedInput)){
                    Vector2<int> shift = Vector2<int>(inputPosPadded.x, inputPosPadded.y) - kernelPos;
                    float weight = kernels[c].getCell(shift.x, shift.y, inputPosPadded.z);
                    Vector3<int> outputPos (kernelPos.x / strideX, kernelPos.y / strideY, c);
                    result += weight * activationFunction.derive(beforeActivation->getCell(outputPos)) * network->getChain(__id + 1, outputPos);
                }
            }
        }
    }
    setMemo(inputPos, result);
    return result;
}

float cn::ConvolutionLayer::diffWeight(int weightID) {
    int kSize = kernelSize.multiplyContent();
    Bitmap<float> paddedInput = Utils::addPadding(*network->getInput(__id), paddingX, paddingY);
    Vector3<int> weightPos = kernels[weightID / kSize].indexToVector(weightID % kSize);
    int kID = weightID / kSize;

    float result = 0;
    for(int y = 0; y < outputSize.y - kernelSize.y; y++){
        for(int x = 0; x < outputSize.x - kernelSize.x; x++){
            Vector3<int> inputPos = Vector3<int>(x * strideX, y * strideY, 0) + weightPos;
            float inputValue = paddedInput.getCell(inputPos);
            Vector3<int> nextPos (x, y, kID);
            result += inputValue * activationFunction.derive(beforeActivation->getCell(nextPos)) * network->getChain(__id + 1, nextPos);
        }
    }
    return result;
}

int cn::ConvolutionLayer::weightsCount() const {
    return kernelSize.multiplyContent() * kernelsCount;
}

std::vector<float> cn::ConvolutionLayer::getWeightsGradient() {
    std::vector<float> gradient(weightsCount());
    for(int i = 0; i < weightsCount(); i ++){
        gradient[i] = diffWeight(i);
    }
    return gradient;
}

void cn::ConvolutionLayer::setWeight(int weightID, float value) {
    int kSize = kernelSize.multiplyContent();
    *(kernels[weightID / (kSize)].data() + (weightID % kSize)) = value;
}

float cn::ConvolutionLayer::getWeight(int weightID) const {
    int kSize = kernelSize.multiplyContent();
    return *(kernels[weightID / (kSize)].data() + (weightID % kSize));
}

std::vector<float> cn::ConvolutionLayer::getBiasesGradient() {
    return std::vector<float>(kernelsCount, 0);
}

void cn::ConvolutionLayer::setBias(int kernelID, float value) {
    biases[kernelID] = value;
}

float cn::ConvolutionLayer::getBias(int kernelID) const {
    return biases[kernelID];
}

int cn::ConvolutionLayer::biasesCount() const {
    return kernelsCount;
}

