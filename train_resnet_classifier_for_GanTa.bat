
python train_resnet_classifier_for_GanTa.py ^
--action train ^
--data_dir E:\ganta_patch_classification ^
--train_subset train2 ^
--val_subset val1 ^
--save_root E:\ganta_patch_cls_results ^
--netname resnet50 ^
--batchsize 8 ^
--lr 0.001 ^
--num_epochs 100

python train_resnet_classifier_for_GanTa.py ^
--action test ^
--data_dir E:\ganta_patch_classification ^
--train_subset train2 ^
--val_subset val1 ^
--test_subset val1 ^
--save_root E:\ganta_patch_cls_results ^
--netname resnet50 ^
--batchsize 8 ^
--lr 0.001 ^
--num_epochs 100