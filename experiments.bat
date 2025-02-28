REM Deney_1
REM python train.py --epochs 300 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgb --device cuda --exp_name rgb --seed 42 --patience 15 --project Deney_1
REM python train.py --epochs 300 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type ir --device cuda --exp_name ir --seed 42 --patience 15 --project Deney_1

REM git checkout fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_3x3
REM python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_3x3 --seed 42 --patience 15 --project Deney_1

REM git checkout fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_1x1
REM python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_1x1 --seed 42 --patience 15 --project Deney_1

REM git checkout fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_3x3
REM python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_3x3 --seed 42 --patience 15 --project Deney_1

REM git checkout fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_1x1
REM python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_1x1 --seed 42 --patience 15 --project Deney_1


REM  Deney_2
REM git checkout master
REM python train.py --epochs 300 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbcatir --device cuda --exp_name rgbcatir --seed 42 --patience 30 --project Deney_2 --in_ch_1 4
 
REM git checkout fuse_max_bottleneck_cat_bottleneck_3x3
REM python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_max_bottleneck_cat_bottleneck_3x3 --seed 42 --patience 30 --project Deney_2 --in_ch_1 3 --in_ch_2 3
 
REM git checkout fuse_max_bottleneck_max
REM python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_max_bottleneck_max --seed 42 --patience 30 --project Deney_2 --in_ch_1 3 --in_ch_2 3
 
REM git checkout master
REM python train.py --epochs 300 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgb --device cuda --exp_name rgb --seed 42 --patience 30 --project Deney_2 --in_ch_1 3
REM python train.py --epochs 300 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type ir --device cuda --exp_name ir --seed 42 --patience 30 --project Deney_2 --in_ch_1 3

REM Deney_3
REM git checkout master
REM python train.py --epochs 300 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgb --device cuda --exp_name rgb --seed 42 --patience 30 --project Deney_3 --in_ch_1 3

REM python train.py --epochs 300 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type ir --device cuda --exp_name ir --seed 42 --patience 30 --project Deney_3 --in_ch_1 3

REM python train.py --epochs 300 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbcatir --device cuda --exp_name rgbcatir --seed 42 --patience 30 --project Deney_3 --in_ch_1 4

REM git checkout fuse_add_bottleneck_add
REM python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_add_bottleneck_add --seed 42 --patience 30 --project Deney_3 --in_ch_1 3 --in_ch_2 3

REM git checkout fuse_max_bottleneck_max
REM python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_max_bottleneck_max --seed 42 --patience 30 --project Deney_3 --in_ch_1 3 --in_ch_2 3

REM git checkout fuse_5x5_fuse_cat_bottleneck_cat_bottleneck_5x5
REM python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_5x5_fuse_cat_bottleneck_cat_bottleneck_5x5 --seed 42 --patience 30 --project Deney_3 --in_ch_1 3 --in_ch_2 3

REM git checkout fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_3x3
REM python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_3x3 --seed 42 --patience 30 --project Deney_3 --in_ch_1 3 --in_ch_2 3

REM git checkout fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_3x3
REM python train.py --epochs 300 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_3x3 --seed 42 --patience 30 --project Deney_3 --in_ch_1 3 --in_ch_2 3


REM Deney_4
git checkout master
python train.py --epochs 230 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgb --device cuda --exp_name rgb --seed 42 --patience 50 --project Deney_4 --in_ch_1 3 --aug

python train.py --epochs 230 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type ir --device cuda --exp_name ir --seed 42 --patience 50 --project Deney_4 --in_ch_1 3 --aug

python train.py --epochs 230 --batch_size 8 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbcatir --device cuda --exp_name rgbcatir --seed 42 --patience 50 --project Deney_4 --in_ch_1 4 --aug

git checkout fuse_add_bottleneck_add
python train.py --epochs 230 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_add_bottleneck_add --seed 42 --patience 50 --project Deney_4 --in_ch_1 3 --in_ch_2 3 --aug

git checkout fuse_max_bottleneck_max
python train.py --epochs 230 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_max_bottleneck_max --seed 42 --patience 50 --project Deney_4 --in_ch_1 3 --in_ch_2 3 --aug

git checkout fuse_5x5_fuse_cat_bottleneck_cat_bottleneck_5x5
python train.py --epochs 230 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_5x5_fuse_cat_bottleneck_cat_bottleneck_5x5 --seed 42 --patience 50 --project Deney_4 --in_ch_1 3 --in_ch_2 3 --aug

git checkout fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_3x3
python train.py --epochs 230 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_1x1_fuse_cat_bottleneck_cat_bottleneck_3x3 --seed 42 --patience 50 --project Deney_4 --in_ch_1 3 --in_ch_2 3 --aug

git checkout fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_3x3
python train.py --epochs 230 --batch_size 4 --learning_rate 1e-5 --weight_decay 1e-8 --momentum 0.999 --scale 0.5 --n_classes 12 --train_type rgbir --device cuda --exp_name rgbir_fuse_3x3_fuse_cat_bottleneck_cat_bottleneck_3x3 --seed 42 --patience 50 --project Deney_4 --in_ch_1 3 --in_ch_2 3 --aug
