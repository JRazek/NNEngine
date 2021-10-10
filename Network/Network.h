//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_NETWORK_H
#define NEURALNETLIBRARY_NETWORK_H

#include <vector>
#include <random>
#include "../Utils/Utils.h"
#include "layers/interfaces/Learnable.h"
#include "layers/OutputLayer/OutputLayer.h"
#include "../Utils/interfaces/JSONEncodable.h"
#include <memory>

namespace cn {
    class InputLayer;
    class RecurrentLayer;
    class Network : public JSONEncodable{
    private:

        friend class Optimizer;
        int seed;

        /**
         * what the dimensions of the byte array is after being normalized and sampled
         */
        Vector3<int> inputSize;
        std::default_random_engine randomEngine;
        bool CUDAAccelerate;
    protected:
        std::vector<std::unique_ptr<Layer>> layers;
        std::vector<Learnable *> learnableLayers;

        cn::InputLayer *inputLayer = nullptr;

        OutputLayer *outputLayer = nullptr;

    public:

        void appendConvolutionLayer(Vector2<int> kernelSize, int kernelsCount, Vector2<int> stride = {1, 1}, Vector2<int> padding = {0, 0});
        void appendFFLayer(int neuronsCount);
        void appendFlatteningLayer();
        void appendBatchNormalizationLayer();
        void appendMaxPoolingLayer(Vector2<int> kernelSize);
        void appendRecurrentLayer();
        void appendReLULayer();
        void appendSigmoidLayer();
        void appendRecurrentLayer(std::unique_ptr<RecurrentLayer> &&recurrentLayer);


        void feed(Tensor<double> bitmap);
        void feed(const Tensor<cn::byte> &bitmap);

        /**
         * when network structure is ready - CPURun this function.
         */
        void ready();



        Network(const cn::JSON &json);


        /**
         * randomly initialized all the weights and biases of the network
         */
        void initRandom();

        /**
         * the recurrentLayer is a complex layer and it has to be built by the user. Despite that it also needs input size in constructor which is known by the network only.
         * Therefore use this to construct a Recurrent Layer
         */
         std::unique_ptr<RecurrentLayer> createRecurrentLayer();

        /**
         *
         * @param layerID
         * @return the size vector of nth layer's output
         */
        Vector3<int> getOutputSize(int layerID) const;

        /**
         *
         * @param layerID
         * @return returns input size for each layer
         */
        Vector3<int> getInputSize(int layerID) const;

        void resetState();

        const Tensor<double> &getInput(int layerID, int time) const;
        const Tensor<double> &getNetworkOutput(int time) const;
        const Tensor<double> &getOutput(int layerID, int time) const;

        JSON jsonEncode() const override;

        OutputLayer &getOutputLayer();


        /**
         * input size for feed
         * @param w width
         * @param h height
         * @param d depth
         * @param _seed - seed for random engine
         * if the first layer is image - set all the properties.
         * In case of using only FFLayers - set height and depth to 1.
         */

        Network(int w, int h, int d, int _seed = 1, bool CUDAAccelerate = 0);


        explicit Network(cn::Vector3<int> _inputSize, int _seed = 1, bool CUDAAccelerate = 0);

        Network (const Network &network) = delete;
        Network &operator=(const Network &network) = delete;
        Network (Network &&network);
        Network &operator=(Network &&network);
        int layersCount() const;
        bool isCudaAccelerate() const;

        static void linkLayers(std::vector<std::unique_ptr<Layer>> &layers);
    };
}


#endif //NEURALNETLIBRARY_NETWORK_H
