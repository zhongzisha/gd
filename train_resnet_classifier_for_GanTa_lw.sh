export CUDA_VISIBLE_DEVICES=1
BS=64
LR=0.0025
NUM_EPOCHS=100

python train_resnet_classifier_for_GanTa.py \
--action train \
--data_dir /media/ubuntu/Data/ganta_patch_classification \
--train_subset train2 \
--val_subset val1 \
--save_root /media/ubuntu/Data/ganta_patch_cls_results \
--netname resnet50 \
--batchsize ${BS} \
--lr ${LR} \
--num_epochs ${NUM_EPOCHS} \
--loss_weights '0.3, 0.7' \
--version v0


#python train_resnet_classifier_for_GanTa.py \
#--action test \
#--data_dir /media/ubuntu/Data/ganta_patch_classification \
#--train_subset train2 \
#--val_subset val1 \
#--test_subset val1 \
#--save_root /media/ubuntu/Data/ganta_patch_cls_results \
#--netname resnet50 \
#--batchsize ${BS} \
#--lr ${LR} \
#--num_epochs ${NUM_EPOCHS} \
#--loss_weights '' \
#--version v0