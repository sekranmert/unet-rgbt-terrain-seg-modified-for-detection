python train.py --epochs 300 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgb --device cuda --exp_name rgb --seed 42 --patience 15 --project Deney_1
python train.py --epochs 300 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type ir --device cuda --exp_name ir --seed 42 --patience 15 --project Deney_1

git checkout fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_3x3
python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_3x3 --seed 42 --patience 15 --project Deney_1

git checkout fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_1x1
python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_1x1 --seed 42 --patience 15 --project Deney_1

git checkout fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_3x3
python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_3x3 --seed 42 --patience 15 --project Deney_1

git checkout fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_1x1
python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_1x1 --seed 42 --patience 15 --project Deney_1
