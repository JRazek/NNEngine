//
// Created by jrazek on 27.07.2021.
//
#include "ConvolutionLayer.h"
#include "CUDAConvolutionLayer.cuh"
#include "../../Network.h"
#include <future>

void cn::ConvolutionLayer::CPURun(const Bitmap<double> &_input) {
    if(inputSize != _input.size()){
        throw std::logic_error("CLayer fed with wrong _input size!");
    }
    std::vector<std::future<Bitmap<double>>> kernelThreads;
    auto getConvolved = [this](const Bitmap<double> &input, const cn::Bitmap<double> &kernel){
        return Utils::sumBitmapLayers(Utils::convolve(kernel, input, padding.x, padding.y, stride.x, stride.y));
    };
    for(int i = 0; i < kernelsCount; i ++){
        kernelThreads.push_back(std::async(getConvolved, _input, kernels[i]));
    }
    Bitmap<double> result(outputSize);
    for(int i = 0; i < kernelsCount; i ++){
        result.setLayer(i, kernelThreads[i].get().data());
    }
    output = std::make_unique<Bitmap<double>>(std::move(result));
}

void cn::ConvolutionLayer::CUDARun(const cn::Bitmap<double> &_input) {
    if(inputSize != _input.size()){
        throw std::logic_error("CLayer fed with wrong _input size!");
    }
    output = std::make_unique<Bitmap<double>>(CUDAConvolutionLayer::CUDARun(*this, _input));
}

void cn::ConvolutionLayer::randomInit(std::default_random_engine &randomEngine) {
    std::uniform_real_distribution<> dis(0, 1);

    for(auto &k : kernels){
        for(auto it = k.data(); it != k.data() + k.w() * k.h() * k.d(); ++it){
            double tmp = dis(randomEngine);
            *it = tmp;
        }
    }
    for(auto &b : biases){
        //todo test differentiation
        b = 0;
    }
}

double cn::ConvolutionLayer::getChain(const Vector3<int> &inputPos) {
    if(getMemoState(inputPos)){
        return getMemo(inputPos);
    }
    Bitmap<double> paddedInput = Utils::addPadding(*getInput().get(), padding.x, padding.y);

    auto validPos = [this](const Vector2<int> &kernelPos, const Bitmap<double> &bitmap){
        return kernelPos.x >= 0 && kernelPos.y >= 0 && kernelPos.x + kernelSize.x - 1 < bitmap.w() && kernelPos.y + kernelSize.y - 1 < bitmap.h();
    };

    double result = 0;
    Vector2<int> paddedInputPos(padding.x + inputPos.x, padding.y + inputPos.y);
    for(int z = 0; z < kernelsCount; z++) {
        for (int y = 0; y < kernelSize.y; y++) {
            for (int x = 0; x < kernelSize.x; x++) {
                Vector2<int> kernelPos(paddedInputPos.x - x, paddedInputPos.y - y);
                if (validPos(kernelPos, paddedInput)) {
                    Vector3<int> weightPos(paddedInputPos.x - kernelPos.x, paddedInputPos.y - kernelPos.y, inputPos.z);
                    float weight = kernels[z].getCell(weightPos);
                    Vector3<int> outputPos(kernelPos.x / stride.x, kernelPos.y /stride.y, z);
                    result += weight * nextLayer->getChain(outputPos);
                }
            }
        }
    }
    setMemo(inputPos, result);
    return result;
}

double cn::ConvolutionLayer::diffWeight(int weightID) {
    int kSize = kernelSize.multiplyContent();
    Bitmap<double> paddedInput = Utils::addPadding(*getInput().get(), padding.x, padding.y);
    Vector3<int> weightPos = kernels[weightID / kSize].indexToVector(weightID % kSize);
    int kID = weightID / kSize;

    double result = 0;
    for(int y = 0; y < outputSize.y - kernelSize.y; y++){
        for(int x = 0; x < outputSize.x - kernelSize.x; x++){
            Vector3<int> inputPos = Vector3<int>(x * stride.x, y * stride.y, 0) + weightPos;
            double inputValue = paddedInput.getCell(inputPos);
            Vector3<int> outputPos (x, y, kID);
            result += inputValue * nextLayer->getChain(outputPos);
        }
    }
    return result;
}


double cn::ConvolutionLayer::diffBias(int biasID) {
    double res = 0;
    for(int y = 0; y < outputSize.y; y++){
        for(int x = 0; x < outputSize.x; x++){
            res += getChain({x, y, biasID});
        }
    }
    return res;
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
    return *(kernels[weightID / (kSize)].dataConst() + (weightID % kSize));
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
    structure["input_size"] = inputSize.jsonEncode();
    structure["kernels"] = std::vector<JSON>();
    structure["stride"] = stride.jsonEncode();
    structure["padding"] = padding.jsonEncode();
    structure["kernel_size"] = kernelSize.jsonEncode();
    structure["learnable"] = true;
    for(int i = 0; i < kernelsCount; i ++){
        JSON s;
        s["weights"] = kernels[i].jsonEncode();
        s["bias"] = biases[i];
        structure["kernels"].push_back(s);
    }
    return structure;
}

cn::ConvolutionLayer::ConvolutionLayer(const JSON &json) :
        ConvolutionLayer(json.at("id"),
                         json.at("input_size"),
                         json.at("kernel_size"),
                         json.at("kernels").size(),
                         json.at("stride"),
                         json.at("padding")) {

    for(u_long i = 0; i < json.at("kernels").size(); i ++){
        auto k = json.at("kernels")[i];
        kernels[i] = k.at("weights");
        biases[i] = k.at("bias");
    }
}

cn::ConvolutionLayer::ConvolutionLayer(int _id, Vector3<int> _inputSize, Vector2<int> _kernelSize, int _kernelsCount,
                                       Vector2<int> _stride, Vector2<int> _padding) :
Learnable(_id, _inputSize, _kernelsCount),
kernelSize(_kernelSize.x, _kernelSize.y, _inputSize.z),
kernelsCount(_kernelsCount),
padding(_padding), stride(_stride),
biases(_kernelsCount){
    if(inputSize.x < kernelSize.x || inputSize.y < kernelSize.y){
        throw std::logic_error("kernel must not be larger than input!");
    }
    kernels.reserve(_kernelsCount);

    for(int i = 0; i < _kernelsCount; i ++){
        kernels.emplace_back(kernelSize);
    }

    int oX = Utils::afterConvolutionSize(kernelSize.x, inputSize.x, padding.x, stride.x);
    int oY = Utils::afterConvolutionSize(kernelSize.y, inputSize.y, padding.y, stride.y);
    int oZ = kernelsCount;
    outputSize = Vector3<int>(oX, oY, oZ);
}

std::unique_ptr<cn::Layer> cn::ConvolutionLayer::getCopyAsUniquePtr() const {
    return std::make_unique<ConvolutionLayer>(*this);
}

void cn::ConvolutionLayer::CUDAAutoGrad() {
    CUDAConvolutionLayer::CUDAAutoGrad(*this);
}
