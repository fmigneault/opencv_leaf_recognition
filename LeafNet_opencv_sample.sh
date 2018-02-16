#!/usr/bin/env sh
# ==========
# config options
# ==========
base_dir=$(pwd)
leafnet_path=$base_dir/install/bin/LeafNet_predict
leafnet_eval="/home/nvidia/leaf_deep_learning/plant_samples/Acer_palmatum0.jpg"

# ==========
# model options
# ==========
# models: Flavia | Foliage | LeafSnap
model_name=LeafSnap
models_dir=$base_dir/../LeafNet/LeafNet_Testing/Model
model_proto=$models_dir/$model_name/$model_name.prototxt
model_caffe=$models_dir/$model_name/$model_name.caffemodel
model_mean=$models_dir/$model_name/${model_name}_mean.npy
model_labels=$models_dir/$model_name/id.txt

# ==========
# command
# ==========
cmd='$leafnet_path
    -image=$leafnet_eval -model=$model_caffe -proto=$model_proto -mean=$model_mean -labels=$model_labels'
cmd_txt=$(eval "echo $cmd")

# ==========
# summary and call
# ==========
echo "path:   [$leafnet_path]"
echo "eval:   [$leafnet_eval]"
echo "model:"
echo "  path:     [$models_dir]"
echo "  name:     [$model_name]"
echo "  caffe:    [$model_caffe]"
echo "  proto:    [$model_proto]"
echo "  mean:     [$model_mean]"
echo "  labels:   [$model_labels]"
echo ""
echo "cmd:"
echo "[$cmd_txt]"
echo ""
eval $cmd

