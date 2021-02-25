#pragma once

struct Net;

struct Layer{
    const int idInNet;
    const Net * net;
    Layer(int id, Net * net):idInNet(id), net(net){}
    virtual ~Layer() = default;
};