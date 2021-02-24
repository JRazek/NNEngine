#pragma once

struct Layer{
    const int idInNet;
    Layer(int id):idInNet(id){}
    virtual ~Layer() = default;
};