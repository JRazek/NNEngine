#pragma once

#include "../../Net.h"

struct Layer{
    const int idInNet;
    //Net * net;
    Layer(int id):idInNet(id){}
    virtual ~Layer() = default;
};