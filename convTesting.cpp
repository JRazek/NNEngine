//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/dataStructures/Bitmap.h"
#include "Utils/Utils.h"
#include "Network/Network.h"
#include "LearningModels/Backpropagation.h"
#include <opencv2/opencv.hpp>
#include "Utils/Files/CSVReader.h"
#include "Utils/Files/ImageRepresentation.h"

int main(){
    cn::Network network(28, 28, 1, 443);

    ReLU reLu;
    Sigmoid sigmoid;

    cn::Backpropagation backpropagation(network,  0.01, 1);

    const int outputSize = 10;
    network.appendFlatteningLayer();
    network.appendFFLayer(16, sigmoid);
    network.appendFFLayer(16, sigmoid);
    network.appendFFLayer(outputSize, sigmoid);
    network.initRandom();
    network.ready();



    CSVReader csvReader("/home/jrazek/CLionProjects/ConvolutionalNetLib/metadata.csv", ';');
    csvReader.readContents();
    auto &contents = csvReader.getContents();
    std::vector<ImageRepresentation> imageRepresentations;
    imageRepresentations.reserve(contents.size());
    for(auto &c : contents){
        std::string path = c[0];
        std::string value = c[1];
        imageRepresentations.emplace_back(path, value);
    }

    cn::Bitmap<double> target (outputSize, 1, 1);
    for(int i = 0; i < outputSize; i ++){
        target.setCell(i, 0, 0, 0);
    }
    auto getBest = [](const cn::Bitmap<double> &output){
        int best = 0;
        for (int j = 0; j < output.w(); ++j) {
            if(output.getCell(j, 0, 0) > output.getCell(best, 0, 0)){
                best = j;
            }
        }
        return best;
    };

    std::shuffle(imageRepresentations.begin(), imageRepresentations.end(), std::default_random_engine(1));

    int correctCount = 0;
    int resetRate = 1000;
    for(int k = 0; k < 1000; k ++)
        for(int i = 0; i < imageRepresentations.size(); i ++) {
            ImageRepresentation &imageRepresentation = imageRepresentations[i];
            cv::Mat mat = cv::imread(imageRepresentation.path);
            cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);
            bitmap = cn::Utils::average3Layers(bitmap);
            int numVal = std::stoi(imageRepresentation.value);
            target.setCell(numVal, 0, 0, 1);

            network.feed(bitmap);
            backpropagation.propagate(target);

            int best = getBest(network.getNetworkOutput());
            if(best == numVal){
                correctCount ++;
            }

            if(!((i + 1) % resetRate)){
                std::cout<<i<<": "<<backpropagation.getError(target)<<"\n";
                std::cout<<"ACCURACY: "<< (double)correctCount / double(resetRate) * 100<<"%\n";
                correctCount = 0;
            }

            target.setCell(numVal, 0, 0, 0);
        }

    return 0;
}