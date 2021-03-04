#pragma once
#include <vector>

struct Net;

struct Layer{
    const int idInNet;
    const Net * net;
    std::vector<float> outputVector;
    Layer(int id, Net * net):idInNet(id), net(net){}
    virtual ~Layer(){};
};