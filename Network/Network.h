//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_NETWORK_H
#define NEURALNETLIBRARY_NETWORK_H

#include <vector>
#include "../Utils/Bitmap.h"
#include "../Utils/Utils.h"

namespace cn {
    class Layer;

    class FFLayer;

    class ConvolutionLayer;

    class Network {
    protected:
        std::vector<Layer *> layers;
        friend class Layer;

        friend ConvolutionLayer;


        void appendLayer(Layer * layer);
    public:

        void appendConvolutionLayer(int kernelX, int kernelY, int kernelZ, int kernelsCount, int paddingX = 0,
                                    int paddingY = 0);


        /**
         * what the dimensions of the byte array is after being normalized and sampled
         */
        const int inputDataWidth;
        const int inputDataHeight;
        const int inputDataDepth;

        /**
         * if first layer is convolution - use this feed method
         * @param data - dataP to convolve
         * @param w - width of tensor
         * @param h - height of tensor
         * @param d - depth of tensor
         */

        void feed(const byte *input);

        /**
         * use this if and only if first layer is convolution
         * @param data
         */

        void feed(const Bitmap<float> &bitmap);


        const std::vector<Layer *> * getLayers();

        /**
         * input size for feed
         * @param w width
         * @param h height
         * @param d depth
         *
         * if the first layer is image - set all the properties.
         * In case of using only FFLayers - set height and depth to 1.
         */

        Network(int w, int h, int d);

        ~Network();
    };
}


#endif //NEURALNETLIBRARY_NETWORK_H
