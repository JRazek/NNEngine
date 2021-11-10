//
// Created by jrazek on 27.07.2021.
//

#include "FFLayer.h"
#include "../../Network.h"


cn::FFLayer::FFLayer(const JSON &json) :
        FFLayer(json.at("input_size"), json.at("neurons_count")) {
    std::vector<double> w = json.at("weights");
    std::vector<double> b = json.at("biases");
    weights = w;
    biases = b;
}


cn::FFLayer::FFLayer(Vector3<int> _inputSize, int _neuronsCount) :
        Learnable(_inputSize),
        biases(_neuronsCount),
        neuronsCount(_neuronsCount){

    if(inputSize.x < 1 || inputSize.y != 1 || inputSize.z != 1){
        throw std::logic_error("There must be a vector output layer before FFLayer!");
    }
    weights = std::vector<double>(neuronsCount * inputSize.x);
    outputSize = Vector3<int> (neuronsCount, 1, 1);
}

void cn::FFLayer::CPURun(const Tensor<double> &_input) {
    if(_input.size() != inputSize){
        throw std::logic_error("_input bitmap to ff layer must be a normalized vector type!");
    }
    int weightsPerNeuron = weightsCount() / neuronsCount;

    Tensor<double> result(neuronsCount, 1, 1);
    for(int i = 0; i < neuronsCount; i ++){
        double sum = 0;
        for(int j = 0; j < weightsPerNeuron; j ++){
            sum += _input.getCell(j, 0, 0) * weights.at(i * weightsPerNeuron + j);
        }
        result.setCell(i, 0, 0, sum);
    }
    output.emplace_back(Tensor<double>(std::move(result)));
    addMemoLayer();
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

double cn::FFLayer::getChain(const Vector4<int> &inputPos) {
    if(inputPos.x < 0 || inputPos.y != 0 || inputPos.z != 0){
        throw std::logic_error("wrong chain request!");
    }
    if(getMemoState(inputPos)){
        return getMemo(inputPos);
    }
    int weightsPerNeuron = inputSize.x;

    double res = 0;
    for(int i = 0; i < neuronsCount; i ++){
        res += weights.at(i * weightsPerNeuron  + inputPos.x) * nextLayer->getChain({i, 0, 0, inputPos.t});
    }

    setMemo(inputPos, res);
    return res;
}

double cn::FFLayer::diffWeight(int weightID) {
    int neuron = weightID / inputSize.x;
    const Tensor<double> &input = getInput(getTime() - 1);
    double result = 0;
    for(int t = 0; t < output.size(); t++){
        result += input.getCell(weightID % inputSize.x, 0, 0) * nextLayer->getChain({neuron, 0, 0, t});
    }
    return result;
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
    double result = 0;
    for(int t = 0; t < output.size(); t++){
        result += nextLayer->getChain({neuronID, 0, 0, t});
    }
    return result;
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
    structure["type"] = "ffl";
    structure["weights"] = weights;
    structure["input_size"] = inputSize.jsonEncode();
    structure["neurons_count"] = neuronsCount;
    structure["biases"] = biases;
    structure["learnable"] = true;

    return structure;
}

std::unique_ptr<cn::Layer> cn::FFLayer::getCopyAsUniquePtr() const noexcept{
    return std::make_unique<FFLayer>(*this);
}

std::vector<double *> cn::FFLayer::getWeightsByRef() {
    std::vector<double *> res(weights.size());
    for(u_int i = 0; i < weights.size(); i ++){
        auto &w = weights[i];
        res[i] = &w;
    }
    return res;
}

std::vector<double *> cn::FFLayer::getBiasesByRef() {
    std::vector<double *> res(biases.size());
    for(u_int i = 0; i < biases.size(); i ++){
        auto &b = biases[i];
        res[i] = &b;
    }
    return res;
}

std::unique_ptr<cn::Layer> cn::FFLayer::reproduce(const cn::Layer *netT, int seed) const {
    const FFLayer* net2 = dynamic_cast<const FFLayer *>(netT);
    std::unique_ptr<FFLayer> child = std::make_unique<FFLayer>(*this);

    std::default_random_engine randomEngine(seed);
    std::uniform_int_distribution<> dist(0, 1);
    for(auto i = 0; i < weightsCount(); i ++){
        bool rand = dist(randomEngine);
        if(rand)
            child->weights[i] = net2->weights[i];
    }
    return child;
}
