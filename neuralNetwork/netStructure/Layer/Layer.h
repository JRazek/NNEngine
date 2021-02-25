#pragma once

struct Net;

struct Layer{
    const int idInNet;
    Net * net;
    Layer(int id):idInNet(id){}
    virtual ~Layer() = default;
};