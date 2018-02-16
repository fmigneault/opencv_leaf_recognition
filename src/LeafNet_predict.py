"""
Copyright (c) 2016, Pierre Barré
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os, sys
usr = os.environ['LOGNAME']
sys.path.append('/home/' + usr + '/caffe/install/python')
import caffe
import caffe.proto.caffe_pb2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os


class classify_data():
    """
    Classify an image or a set of images
    """

    def __init__(self):
        self.option = ""
        self.path = []
        self.net_param = []
        self.conf_matr = None
        self.id_dic = {}

    def set_id_dic(self):
        """
        make a dictionary from label to name for classification
        :return:
        """
        id_data = open(self.path[5])
        for line in id_data:
            name, label = line.split(" ")
            self.id_dic[int(label)] = name
        id_data.close()

    def set_option(self, opt):
        """
        Set the option of the classification after users answers
        :param opt: dataset
        :return:
        """
        self.option = opt

    def set_path(self, o_path, path_to_m):
        """
        create a path database
        :param o_path: path of the dataset
        :return:
        """
        path_database = []
        if self.option == "other":
            main_path = path_to_m.split("/snapshot")[0]
            path_database.append(path_to_m)
            file_path = ["/lf.prototxt", "/mean.npy", "/normalization/", "/data/validation.txt", "/data/id.txt"]
            for i in range(0,5):
                path_database.append(main_path + file_path[i])
        else:
            file = [".caffemodel", ".prototxt", "_mean.npy"]
            for i in range(0,3):
                path_database.append(o_path + self.option + "/" + self.option + file[i])
            path_database.append(o_path + self.option + "/" + "validation/")
            path_database.append(o_path + self.option + "/" + "validation.txt")
            path_database.append(o_path + self.option + "/" + "id.txt")
        self.path = path_database

    def set_net(self, c_or_g):
        """
        Create the LeafNet
        :return:
        """
        if c_or_g == "GPU":
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        return caffe.Net(self.path[1], self.path[0], caffe.TEST)

    def set_transformer(self):
        """
        Prepare the images for the Net
        :return:
        """
        mean_dataset = np.load(self.path[2])
        mean_dataset = mean_dataset.mean(1).mean(1)
        self.net_param[1].set_mean('data', mean_dataset)
        self.net_param[1].set_transpose('data', (2, 0, 1))
        self.net_param[1].set_raw_scale('data', 255)
        self.net_param[1].set_channel_swap('data', (2, 1, 0))

    def set_species_name(self, file):
        """
        Get all species name
        :param file: id.txt file
        :return:
        """
        species = {}
        f = open(file, "r")
        for line in f:
            line_split = line.split(" ")
            species[int(line_split[1])] = line_split[0]
        return species

    def initialize_net(self, cg):
        """
        iniatlize the net for the classification
        :param cg: cpu or gpu
        :return:
        """
        self.net_param.append(self.set_net(cg))
        transformer = caffe.io.Transformer({'data': self.net_param[0].blobs['data'].data.shape})
        self.net_param.append(transformer)
        self.set_transformer()

    def classification_one(self, img, option_n):
        """
        proceed to one image classification
        :param img: path to the image for the classification
        :return:
        """
        self.set_id_dic()
        input_image = caffe.io.load_image(img)
        self.net_param[0].blobs['data'].data[...] = self.net_param[1].preprocess('data', input_image)
        output = self.net_param[0].forward()
        output_id = output['prob'][0]
        if option_n:
            for i in output_id.argsort()[::-1][:5]:
                print(self.id_dic[i] , "  " , str(output_id[i]*100) , " % ")
        else:
            result = self.id_dic[output_id.argmax()] + " " + str(output_id[output_id.argmax()]*100) + " % "
            return result

    def create_confusionsmatrix(self, pr_1, t_label):
        """
        Create the confusion matrix from the results of an all classification
        :param pr_1: predictions from the net
        :param t_label: the true label
        :return:
        """
        self.conf_matr = confusion_matrix(pr_1, t_label)

    def get_accuracy(self, pr_1, pr_5, t_label):
        """
        get the accuracy from results
        :param pr_1: predictions top-1
        :param pr_5: prediction top-5
        :param t_label: true label
        :return:
        """
        print('Erkennungsrate Top-1 : ' + str(accuracy_score(t_label, pr_1, normalize=True)))
        print('Erkennungsrate Top-5 : ' + str(accuracy_score(t_label, pr_5, normalize=True)))

    def classification_directory(self, image_directory):
        result = open(image_directory + "/results.txt", "w")
        for f in os.listdir(image_directory):
            name, e = f.split(".")
            if e != "jpg":
                continue
            else:
                image_path = (image_directory + "/" + f)
                result.write(name + " " +  self.classification_one(image_path, False) + '\n')

    def classification_all(self):
        """
        make a classification of all images from a dataset
        :return:
        """
        predictions_top1 = []
        predictions_top5 = []
        testing_Label = []
        test_data = open(self.path[4], "r")
        progress = 0
        progress_all = 0
        for im in test_data:
           progress_all += 1
        test_data.close()
        test_data = open(self.path[4], "r")
        for line in test_data:
            if progress%50 == 0:
                print("Progress : ", progress, "/", progress_all)
            img, label = line.split(" ")
            testing_Label.append(int(label))
            input_image = caffe.io.load_image(self.path[3] + img)
            self.net_param[0].blobs['data'].data[...] = self.net_param[1].preprocess('data', input_image)
            output = self.net_param[0].forward()
            output_prob = output['prob'][0]
            if int(label) in output_prob.argsort()[::-1][:5]:
                predictions_top5.append(int(label))
            else:
                predictions_top5.append(output_prob.argmax())
            predictions_top1.append(output_prob.argmax())
            progress += 1
        test_data.close()
        self.get_accuracy(predictions_top1, predictions_top5, testing_Label)
        self.create_confusionsmatrix(predictions_top1, testing_Label)



