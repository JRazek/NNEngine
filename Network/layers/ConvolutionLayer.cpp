//
// Created by jrazek on 27.07.2021.
//
#include "ConvolutionLayer.h"
#include "../Network.h"
#include <future>
cn::ConvolutionLayer::ConvolutionLayer(int _id, Network &_network, int _kernelSizeX, int _kernelSizeY,
                                       int _kernelsCount, int _strideX, int _strideY, int _paddingX, int _paddingY) :
        cn::Learnable(_id, _network, _kernelsCount),
        kernelSize(_kernelSizeX, _kernelSizeY, network->getInputSize(_id).z),
        kernelsCount(_kernelsCount),
        paddingX(_paddingX),
        paddingY(_paddingY),
        strideX(_strideX),
        strideY(_strideY),
        biases(kernelsCount) {

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
}

cn::Bitmap<double> cn::ConvolutionLayer::run(const Bitmap<double> &input) {
    if(input.size() != network->getInputSize(__id)){
        throw std::logic_error("CLayer fed with wrong input size!");
    }
    std::vector<std::future<Bitmap<double>>> kernelThreads;
    auto getConvolved = [this](const Bitmap<double> &input, const cn::Bitmap<double> &kernel){
        return Utils::sumBitmapLayers(Utils::convolve(kernel, input, paddingX, paddingY, strideX, strideY));
    };
    for(int i = 0; i < kernelsCount; i ++){
        kernelThreads.push_back(std::async(getConvolved, input, kernels[i]));
    }
    Bitmap<double> result(outputSize);
    for(int i = 0; i < kernelsCount; i ++){
        result.setLayer(i, kernelThreads[i].get().data());
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

double cn::ConvolutionLayer::getChain(const Vector3<int> &inputPos) {
    if(getMemoState(inputPos)){
        return getMemo(inputPos);
    }
    Bitmap<double> paddedInput = Utils::addPadding(network->getInput(__id), paddingX, paddingY);

    auto validPos = [this](const Vector2<int> &kernelPos, const Bitmap<double> &bitmap){
        return kernelPos.x >= 0 && kernelPos.y >= 0 && kernelPos.x + kernelSize.x - 1 < bitmap.w() && kernelPos.y + kernelSize.y - 1 < bitmap.h();
    };

    double result = 0;
    Vector2<int> paddedInputPos(paddingX + inputPos.x, paddingY + inputPos.y);
    for(int z = 0; z < kernelsCount; z++) {
        for (int y = 0; y < kernelSize.y; y++) {
            for (int x = 0; x < kernelSize.x; x++) {
                Vector2<int> kernelPos(paddedInputPos.x - x, paddedInputPos.y - y);
                if (validPos(kernelPos, paddedInput)) {
                    Vector3<int> weightPos(paddedInputPos.x - kernelPos.x, paddedInputPos.y - kernelPos.y, inputPos.z);
                    float weight = kernels[z].getCell(weightPos);
                    Vector3<int> outputPos(kernelPos.x / strideX, kernelPos.y /strideX, z);
                    result += weight * network->getChain(__id + 1, outputPos);
                }
            }
        }
    }
    setMemo(inputPos, result);
    return result;
}

double cn::ConvolutionLayer::diffWeight(int weightID) {
    int kSize = kernelSize.multiplyContent();
    Bitmap<double> paddedInput = Utils::addPadding(network->getInput(__id), paddingX, paddingY);
    Vector3<int> weightPos = kernels[weightID / kSize].indexToVector(weightID % kSize);
    int kID = weightID / kSize;

    double result = 0;
    for(int y = 0; y < outputSize.y - kernelSize.y; y++){
        for(int x = 0; x < outputSize.x - kernelSize.x; x++){
            Vector3<int> inputPos = Vector3<int>(x * strideX, y * strideY, 0) + weightPos;
            double inputValue = paddedInput.getCell(inputPos);
            Vector3<int> outputPos (x, y, kID);
            result += inputValue * network->getChain(__id + 1, outputPos);
        }
    }
    return result;
}

int cn::ConvolutionLayer::weightsCount() const {
    return kernelSize.multiplyContent() * kernelsCount;
}

std::vector<double> cn::ConvolutionLayer::getWeightsGradient() {
    std::vector<double> gradient(weightsCount());
    for(int i = 0; i < weightsCount(); i ++){
        gradient[i] = diffWeight(i);
    }
    return gradient;
}

void cn::ConvolutionLayer::setWeight(int weightID, double value) {
    int kSize = kernelSize.multiplyContent();
    *(kernels[weightID / (kSize)].data() + (weightID % kSize)) = value;
}

double cn::ConvolutionLayer::getWeight(int weightID) const {
    int kSize = kernelSize.multiplyContent();
    return *(kernels[weightID / (kSize)].data() + (weightID % kSize));
}

std::vector<double> cn::ConvolutionLayer::getBiasesGradient() {
    return std::vector<double>(kernelsCount, 0);
}

void cn::ConvolutionLayer::setBias(int kernelID, double value) {
    biases[kernelID] = value;
}

double cn::ConvolutionLayer::getBias(int kernelID) const {
    return biases[kernelID];
}

int cn::ConvolutionLayer::biasesCount() const {
    return kernelsCount;
}

cn::JSON cn::ConvolutionLayer::jsonEncode() const{
    JSON structure;
    structure["id"] = __id;
    structure["type"] = "cl";
    structure["kernels"] = std::vector<JSON>();
    for(int i = 0; i < kernelsCount; i ++){
        structure["kernels"].push_back(kernels[i].jsonEncode());
    }
    return structure;
}

