//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/dataStructures/Bitmap.h"
#include "Utils/Utils.h"
#include "Network/Network.h"
#include "Optimizers/MBGD.h"
#include <opencv2/opencv.hpp>
#include "Utils/Files/CSVReader.h"
#include "Utils/Files/ImageRepresentation.h"
int main(){
    cn::Network network(28, 28, 1, 6754);

    ReLU reLu;
    Sigmoid sigmoid;

    cn::MBGD mbgd(network, 0.001, 1);

    const int outputSize = 10;
    network.appendConvolutionLayer(3, 3, 2, reLu);
    network.appendConvolutionLayer(3, 3, 6, reLu);
    network.appendBatchNormalizationLayer();
    network.appendConvolutionLayer(3, 3, 10, reLu);
    network.appendFlatteningLayer();
    network.appendBatchNormalizationLayer();
    network.appendFFLayer(16, sigmoid);
    network.appendFFLayer(16, sigmoid);
    network.appendFFLayer(outputSize, sigmoid);
    network.initRandom();
    network.ready();



//
//    cn::Bitmap<cn::byte> testInput(100, 1, 1);
//    for(int i = 0; i < testInput.w(); i ++){
//        testInput.setCell(i, 0, 0, 255);
//    }
//    network.feed(testInput);



    CSVReader csvReader("/home/user/IdeaProjects/digitRecogniser/dataSet/metadata.csv", ';');
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

    int resetRate = 100;
    int correctCount = 0;
    constexpr int epochsCount = 100;
    for (u_int i = 0; i < imageRepresentations.size() * epochsCount; i++) {
        int n = i % imageRepresentations.size();
        ImageRepresentation &imageRepresentation = imageRepresentations[n];
        cv::Mat mat = cv::imread(imageRepresentation.path);

        cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);
        bitmap = cn::Utils::average3Layers(bitmap);
        int numVal = std::stoi(imageRepresentation.value);
        target.setCell(numVal, 0, 0, 1);
        network.feed(bitmap);
        mbgd.propagate(target);

        int best = getBest(network.getNetworkOutput());
        if (best == numVal) {
            correctCount++;
        }

        if (!((i + 1) % resetRate)) {
            std::cout << i << ": " << mbgd.getError(target) << "\n";
            std::cout << "ACCURACY: " << (double) correctCount / double(resetRate) * 100 << "%\n";
            correctCount = 0;
        }

        target.setCell(numVal, 0, 0, 0);
    }
    return 0;
}




/*
 *
 *          cn::Bitmap<cn::byte> bitmap(mat.cols, mat.rows, mat.channels(), mat.data, 1);
            bitmap = cn::Utils::resize(bitmap, 100, 100);
            bitmap = cn::Utils::average3Layers(bitmap);
            cn::byte *tmp = new cn::byte[bitmap.size().multiplyContent()];
            cn::Utils::convert(bitmap.data(), tmp, bitmap.w(), bitmap.h(), bitmap.d(), 0, 1);

            cv::Mat resized (bitmap.w(), bitmap.h(), CV_8UC(bitmap.d()), tmp);


            cv::imshow("", resized);
            cv::waitKey(10000);
            delete[] tmp;

            return 0;
 * */