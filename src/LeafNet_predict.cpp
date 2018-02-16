#include "class_labels.hpp"
#include "cnpy.h"

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
using namespace std;

const size_t inModelW = 256;
const size_t inModelH = 256;

const char* params
    = "{ help           | false                 | print usage }"
      "{ proto          | model.prototxt        | model configuration }"
      "{ model          | model.caffemodel      | model weights }"
      "{ mean           | mean.npy              | model image mean values }"
      "{ labels         | id.txt                | model class labels }"
      "{ image          |                       | image to predict class }"
      "{ min_confidence | 0.2                   | min confidence threshold }"
      "{ opencl         | false                 | enable OpenCL }"
;

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, params);
    parser.about("Sample to predict leaves from LeafNet model "
                 "[https://leafnet.pbarre.de] trained with either "
                 "of the Flavia, Foliage or LeafSnap datasets.\n");

    if (parser.get<bool>("help") || argc == 1) {
        parser.printMessage();
        return 0;
    }

    cv::String modelConfiguration = parser.get<String>("proto");
    cv::String modelBinary = parser.get<String>("model");
    cv::String modelMeanImagePath = parser.get<String>("mean");
    std::vector<std::string> modelLabels = readClassLabels(parser.get<String>("labels"));
    CV_Assert(!modelConfiguration.empty() && !modelBinary.empty() &&
              !modelMeanImagePath.empty() && modelBinary.size() > 0);

    //! [Initialize network]
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
    //! [Initialize network]

    if (parser.get<bool>("opencl"))
        net.setPreferableTarget(DNN_TARGET_OPENCL);

    if (net.empty()) {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelConfiguration << std::endl;
        std::cerr << "caffemodel: " << modelBinary << std::endl;
        std::cerr << "Models can be downloaded here:" << std::endl;
        std::cerr << "https://leafnet.pbarre.de/LeafNet_beta_0.0.1.zip" << std::endl;
        exit(-1);
    }

    float confidenceThreshold = parser.get<float>("min_confidence");
    CV_Assert(confidenceThreshold >= 0 && confidenceThreshold <= 1);

    String display_title = "LeafNet: Prediction";
    namedWindow(display_title, WINDOW_NORMAL | WINDOW_FREERATIO | CV_GUI_EXPANDED);
    resizeWindow(display_title, inModelW, inModelH);

    std::string img_path = parser.get<std::string>("image");
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        cerr << "Can't load image for prediction from following file: " << endl;
        cerr << img_path << endl;
        waitKey();
        return EXIT_FAILURE;
    }

    if (img.channels() == 4)
        cv::cvtColor(img, img, COLOR_BGRA2BGR);
    cv::resize(img, img, Size(inModelW, inModelH), 0, 0, cv::INTER_NEAREST);

    cnpy::NpyArray npyArr = cnpy::npy_load(modelMeanImagePath);
    cv::Mat meanImage(inModelH, inModelW, CV_32FC3, npyArr.data<float>());

    std::cout << "image: " << img.size() << " x " << img.channels() << ", "
              << img.depth() << ", " << img.elemSize() << std::endl;
    std::cout << "mean:  " << meanImage.size() << " x " << meanImage.channels() << ", "
              << meanImage.depth() << ", " << meanImage.elemSize() << std::endl;
    cv::imshow("input image", img);
    cv::imshow("mean image", meanImage);
    ///cv::subtract(img, meanImage, img, cv::noArray(), CV_32FC3);

    //! [Prepare blob]
    cv::Mat inputBlob = blobFromImage(img, 1.0, cv::Size(inModelW, inModelH), Scalar(),
                                      false, false); //Convert Mat to batch of images
    //! [Prepare blob]

    //! [Set input blob]
    net.setInput(inputBlob); //set the network input
    //! [Set input blob]

    //! [Make forward pass]
    cv::Mat leafPredictions = net.forward(); //compute output
    //! [Make forward pass]

    std::vector<double> layersTimings;
    double freq = getTickFrequency() / 1000;
    double time = net.getPerfProfile(layersTimings) / freq;

    cv::Mat leafPredictionsMat(leafPredictions.size[2], leafPredictions.size[3], CV_32F, leafPredictions.ptr<float>());
    std::cout << "Inference time, ms: " << time << std::endl;

    std::string highestPredictionLbl = "unknown";
    double highestPredictionVal = 0;
    cv::Point highestPredictionLoc;
    cv::minMaxLoc(leafPredictions, nullptr, &highestPredictionVal, nullptr, &highestPredictionLoc);
    if (highestPredictionVal >= confidenceThreshold) {
        highestPredictionLbl = modelLabels[highestPredictionLoc.x];
        std::cout << "Prediction: " << highestPredictionVal << " @ " << highestPredictionLoc.x
                  << " (" << highestPredictionLbl << ")" << std::endl;
    }
    else {
        std::cout << "Prediction (" << highestPredictionVal << " @ " << highestPredictionLoc.x
                  << ") lower than min confidence (" << confidenceThreshold << ")" << std::endl;
        highestPredictionVal = -1;
        highestPredictionLoc = cv::Point(-1, -1);
    }

    std::string label = highestPredictionLbl + ": " + std::to_string(highestPredictionVal);
    int baseLine = 0;
    cv::Size labelSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int left = 0;
    int top = max(0, labelSize.height);
    cv::rectangle(img, Point(left, top - labelSize.height),
                  cv::Point(left + labelSize.width, top + baseLine),
                  cv::Scalar(255, 255, 255), CV_FILLED);
    cv::putText(img, label, cv::Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
    cv::imshow(display_title, img);
    cv::waitKey();

    return EXIT_SUCCESS;
}

