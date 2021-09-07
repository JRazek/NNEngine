//
// Created by jrazek on 27.07.2021.
//

#ifndef NEURALNETLIBRARY_LAYER_H
#define NEURALNETLIBRARY_LAYER_H

#include <vector>
#include <optional>
#include "../../../Utils/dataStructures/Bitmap.h"
#include "../../../Utils/interfaces/JSONEncodable.h"

namespace cn {

    class FFLayer;

    template<typename T>
    struct Vector3;
    class Network;
    class Layer : public JSONEncodable{
    protected:
        Network *network;

        Vector3<int> inputSize;
        Vector3<int> outputSize;

        std::optional<Bitmap<bool>> memoizationStates;
        std::optional<Bitmap<double>> memoizationTable;

        int __id;

    public:
        Layer(int _id, Network &_network);

        /**
         *
         * @param bitmap input to process
         * @return output result bitmap of size specified in getOutputSize()
         */
        virtual Bitmap<double> run(const Bitmap<double> &bitmap) = 0;

        virtual double getChain(const Vector3<int> &inputPos) = 0;

        virtual ~Layer() = default;

        void resetMemoization();
        void setMemo(const Vector3<int> &pos, double val);
        bool getMemoState(const Vector3<int> &pos) const;
        double getMemo(const Vector3<int> &pos) const;
        Vector3<int> getOutputSize() const;

        [[maybe_unused]] int id() const;

        //for development purposes only. To delete in future
        virtual JSON jsonEncode() const override;

        static std::unique_ptr<cn::Layer> fromJSON(Network &network, const cn::JSON &json);
    };
}


#endif //NEURALNETLIBRARY_LAYER_H
