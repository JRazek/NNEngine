#pragma once
#include <vector>

struct Net;

struct Layer{
    const int idInNet;
    const Net * net;
    const int outputVectorSize;
    std::vector<float> outputVector;
    Layer(int id, Net * net, int outputVectorSize):idInNet(id), net(net), outputVectorSize(outputVectorSize){}
    virtual ~Layer(){};
};