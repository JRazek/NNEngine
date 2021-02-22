#pragma once

#include <vector>
#include <FFNeuron.h>

struct Layer{
    const int id;
    Layer(int id, int neuronsCount);
    virtual void f();
};