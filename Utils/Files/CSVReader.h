//
// Created by user on 29.08.2021.
//

#ifndef NEURALNETLIBRARY_CSVREADER_H
#define NEURALNETLIBRARY_CSVREADER_H

#include <fstream>
#include <vector>
class CSVReader {
    std::string filepath;
    std::fstream file;
    char splitter;
    std::vector<std::vector<std::string>> contents;
public:
    CSVReader(const std::string &_filePath, char splitter);
    void readContents();
    static std::vector<std::string> split(const std::string &str, char splitter);
    const std::vector<std::vector<std::string>> &getContents() const;
};


#endif //NEURALNETLIBRARY_CSVREADER_H
