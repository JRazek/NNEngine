//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Network/Network.h"

int main(){
    Network network;
    Bitmap bitmap(100,100,3);
    std::vector<byte> data(100 * 100 * 3, 69);
    std::copy(data.data(), data.data() + data.size(), bitmap.getData());

    byte b = bitmap.getByte(99, 99, 2);
    std::cout<<b;
    return 0;
}