#include "class_labels.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


std::vector<std::string> readClassLabels(std::string filename)
{
    std::vector<std::string> classLabels;

    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }

    std::string label;
    while (!fp.eof())
    {
        std::getline(fp, label);
        if (label.length())
            classLabels.push_back(label);
    }

    fp.close();
    return classLabels;
}

