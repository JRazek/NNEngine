//
// Created by user on 18.08.2021.
//

#ifndef NEURALNETLIBRARY_LEARNABLE_H
#define NEURALNETLIBRARY_LEARNABLE_H

namespace cn {
    struct Learnable {
        virtual void randomInit() = 0;
    };
}

#endif //NEURALNETLIBRARY_LEARNABLE_H
