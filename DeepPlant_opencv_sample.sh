#!/usr/bin/env sh
# ==========
# config options
# ==========
base_dir=$(pwd)
deep_plant_path=$base_dir/install/bin/DeepPlant_predict
deep_plant_eval="/home/nvidia/leaf_deep_learning/plant_samples/Acer_palmatum0.jpg"

# ==========
# model options
# ==========
# model_leaf: PT (patch) | WL (whole)
model_leaf=whole
# model_type: model | short
model_type=model
model_name=FinetuneAlexNet_${model_type}_${model_leaf}
models_dir=$base_dir/../deep_plant/deep_plant_Testing/Model
model_proto=$models_dir/$model_name/deploy.prototxt
model_caffe=$models_dir/$model_name/finetune_flickr_style_iter_100000.caffemodel
model_labels=base_dir/../leaf_deep_learning/MalayaKew Dataset/MK/name_of_spesies.txt

# ==========
# command
# ==========
cmd='$deep_plant_path -image=$deep_plant_eval -model=$model_caffe -proto=$model_proto -labels=$model_labels'
cmd_txt=$(eval "echo $cmd")

# ==========
# summary and call
# ==========
echo "path:   [$deep_plant_path]"
echo "eval:   [$deep_plant_eval]"
echo "model:"
echo "  path:     [$models_dir]"
echo "  name:     [$model_name]"
echo "  caffe:    [$model_caffe]"
echo "  proto:    [$model_proto]"
echo "  labels:   [$model_labels]"
echo ""
echo "cmd:"
echo "[$cmd_txt]"
echo ""
eval $cmd
