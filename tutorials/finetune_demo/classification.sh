export CUDA_VISIBLE_DEVICES=4,5

# use this for full finetuning
accelerate launch --config_file tutorials/finetune_demo/ds.yaml \
    tutorials/finetune_demo/classification.py \
    --base_path path to your ptbxl base folder \
    --cache_dir path to cache directory for preprocessed dataset \
    --mode full_finetuning \
    --output_path path to store train log and checkpoint \

# #use this for linear_probing, svm, unsupervised_representation_learning
python3 tutorials/finetune_demo/classification.py \
    --base_path path to your ptbxl base folder \
    --cache_dir path to cache directory for preprocessed dataset \
    --mode linear_probing \
    --output_path path to store train log and checkpoint \