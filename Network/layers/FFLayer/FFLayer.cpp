//
// Created by jrazek on 27.07.2021.
//

#include "FFLayer.h"
#include "../../Network.h"


cn::FFLayer::FFLayer(const JSON &json) :
FFLayer(json.at("id"), json.at("input_size"), json.at("neurons_count")){
    std::vector<double> w = json.at("weights");
    std::vector<double> b = json.at("biases");
    weights = w;
    biases = b;
}


cn::FFLayer::FFLayer(int _id, Vector3<int> _inputSize, int _neuronsCount) :
        Learnable(_id, _inputSize, _neuronsCount),
        biases(_neuronsCount){

    if(inputSize.x < 1 || inputSize.y != 1 || inputSize.z != 1){
        throw std::logic_error("There must be a vector output layer before FFLayer!");
    }
    weights = std::vector<double>(neuronsCount * inputSize.x);
    outputSize = Vector3<int> (neuronsCount, 1, 1);
}

void cn::FFLayer::run(const Bitmap<double> &_input) {
    if(_input.size() != inputSize){
        throw std::logic_error("_input bitmap to ff layer must be a normalized vector type!");
    }
    int weightsPerNeuron = weightsCount() / neuronsCount;

    Bitmap<double> result(neuronsCount, 1, 1);
    for(int i = 0; i < neuronsCount; i ++){
        double sum = 0;
        for(int j = 0; j < weightsPerNeuron; j ++){
            sum += _input.getCell(j, 0, 0) * weights.at(i * weightsPerNeuron + j);
        }
        result.setCell(i, 0, 0, sum);
    }
    output.emplace(std::move(result));
}

void cn::FFLayer::randomInit(std::default_random_engine &randomEngine) {
    std::uniform_real_distribution<> dis(-10, 10);

    for(auto &w : weights){
        w =  dis(randomEngine) * std::sqrt(2.f/inputSize.x);
    }
    for(auto &b : biases){
        b = dis(randomEngine);
    }
}

double cn::FFLayer::getChain(const Vector3<int> &inputPos) {
    if(inputPos.x < 0 || inputPos.y != 0 || inputPos.z != 0){
        throw std::logic_error("wrong chain request!");
    }
    if(getMemoState(inputPos)){
        return getMemo(inputPos);
    }
    int weightsPerNeuron = inputSize.x;

    double res = 0;
    for(int i = 0; i < neuronsCount; i ++){
        res += weights.at(i * weightsPerNeuron  + inputPos.x) * nextLayer->getChain({i, 0, 0});
    }

    setMemo(inputPos, res);
    return res;
}

double cn::FFLayer::diffWeight(int weightID) {
    int neuron = weightID / inputSize.x;
    const Bitmap<double> &input = getInput().value();
    return input.getCell(weightID % inputSize.x, 0, 0) * nextLayer->getChain({neuron, 0, 0});
}

int cn::FFLayer::weightsCount() const {
    return weights.size();
}

std::vector<double> cn::FFLayer::getWeightsGradient() {
    std::vector<double> gradient(weightsCount());
    for(int i = 0; i < weightsCount(); i ++){
        gradient[i] = diffWeight(i);
    }
    return gradient;
}

void cn::FFLayer::setWeight(int weightID, double value) {
    weights[weightID] = value;
}

double cn::FFLayer::getWeight(int weightID) const {
    return weights[weightID];
}

std::vector<double> cn::FFLayer::getBiasesGradient() {
    std::vector<double> gradient(neuronsCount);
    for(int i = 0; i < neuronsCount; i++){
        gradient[i] = diffBias(i);
    }
    return gradient;
}

double cn::FFLayer::diffBias(int neuronID) {
    return nextLayer->getChain({neuronID, 0, 0});
}

void cn::FFLayer::setBias(int neuronID, double value) {
    biases[neuronID] = value;
}

double cn::FFLayer::getBias(int neuronID) const {
    return biases[neuronID];
}

int cn::FFLayer::biasesCount() const {
    return neuronsCount;
}

cn::JSON cn::FFLayer::jsonEncode() const{
    JSON structure;
    structure["id"] = __id;
    structure["type"] = "ffl";
    structure["weights"] = weights;
    structure["input_size"] = inputSize.jsonEncode();
    structure["neurons_count"] = neuronsCount;
    structure["biases"] = biases;
    structure["learnable"] = true;

    return structure;
}

std::unique_ptr<cn::Layer> cn::FFLayer::getCopyAsUniquePtr() const {
    return std::make_unique<FFLayer>(*this);
}
