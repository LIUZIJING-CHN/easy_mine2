# CUDA_VISIBLE_DEVICES=0 python mymain.py \
# --dataset-path /data_25T/lzj/easy/data \
# --dataset miniimagenet \
# --model resnet12 \
# --epochs 0 \
# --manifold-mixup 500 \
# --rotations \
# --cosine \
# --gamma 0.9 \
# --milestones 100 \
# --batch-size 128 \
# --preprocessing ME \
# --n-shots [1,5] \
# --skip-epochs 450 \
# --save-model result/myresnet12_version1.pkl


CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset-path /data/lzj/easy \
--dataset cifarfs \
--model resnet12 \
--epochs 0 \
--load-model /data/lzj/easy/result/1027/res12_64_cifarfs_origin/res12_64_cifarfs_official.pkl1 \
--save-features /data/lzj/easy/save_features/1027/res12_cifarfs_64class/res12_64_cifarfs_official.pt \
--n-shots 1 \
--batch-size 128 \
--preprocessing ME \
--sample-aug 30 

# Test features on miniimagenet using ASY
# CUDA_VISIBLE_DEVICES=0 python main.py \
# --dataset-path /data/lzj/easy/data \
# --dataset miniimagenet \
# --model resnet12 \
# --test-features /data/lzj/easy/save_features/minifeaturesAS_ResNet12_√2_2.pt11 \
# --preprocessing ME \
# --feature-maps 45 \
# --n-shots 1

# Test features on miniimagenet using EASY (nx)
# CUDA_VISIBLE_DEVICES=0 python main.py \
# --dataset-path /data/lzj/easy \
# --dataset miniimagenet \
# --model resnet12 \
# --test-features "/data/lzj/easy/save_features/0818/minifeaturesAS_4.pt1" \
# --preprocessing ME \
# --n-shots 1