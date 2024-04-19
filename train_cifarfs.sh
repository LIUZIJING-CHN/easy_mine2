# CUDA_VISIBLE_DEVICES=3 python main.py \
# --dataset-path /data/lzj/ImageNet \
# --dataset miniimagenet \
# --model resnet12 \
# --epochs 0 \
# --manifold-mixup 7 \
# --load-model result/1018/res12_100class/res12_100class.pkl \
# --rotations \
# --cosine \
# --gamma 0.3 \
# --lr 0.001 \
# --milestones 100 \
# --batch-size 128 \
# --preprocessing ME \
# --n-shots [1,5] \
# --skip-epochs 0 \
# --save-model result/1018/res12_100class/res12_100class_new.pkl

# CUDA_VISIBLE_DEVICES=0 python main.py \
# --dataset-path /data/lzj/ImageNet \
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
# --save-model result/1018/res12_80class/res12_80class_new.pkl

# CUDA_VISIBLE_DEVICES=0 python main.py \
# --dataset-path /data/lzj/easy \
# --dataset cifarfs \
# --model resnet12 \
# --epochs 0 \
# --manifold-mixup 500 \
# --rotations \
# --cosine \
# --gamma 0.9 \
# --lr 0.1 \
# --milestones 100 \
# --batch-size 256 \
# --preprocessing ME \
# --n-shots [1,5] \
# --skip-epochs 480 \
# --save-model /data/lzj/easy/result/1027/res12_64_cifarfs_origin/res12_64_cifarfs_origin.pkl

# CUDA_VISIBLE_DEVICES=0 python main.py \
# --dataset-path /data/lzj/easy \
# --dataset cifarfs \
# --mixup \
# --skip-epochs 300 \
# --batch-size 256 \
# --rotations \
# --preprocessing "PEME" \
# --save-model /data/lzj/easy/result/1027/res12_64_cifarfs_origin/res12_64_cifarfs_PEME_origin.pkl

# python main.py --dataset cifarfs --mixup --rotations --skip-epochs 300 --preprocessing "PEME"
# python main.py --dataset cifarfs --mixup --model wideresnet --feature-maps 16 --skip-epochs 300 --rotations --preprocessing "PEME"
# CUDA_VISIBLE_DEVICES=0 python main.py \
# --dataset-path /data/lzj/easy \
# --dataset cifarfs \
# --mixup \
# --feature-map 16 \
# --model wideresnet \
# --skip-epochs 300 \
# --batch-size 128 \
# --rotations \
# --preprocessing "PEME" \
# --save-model /data/lzj/easy/result/1027/wideresnet_feat16_64_cifarfs_origin/res12_64_cifarfs_origin_2.pkl

CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset-path /data/lzj/easy \
--dataset cifarfs \
--model resnet12 \
--milestones 300 \
--epochs 0 \
--manifold-mixup 1500 \
--cosine \
--gamma 0.9 \
--rotations \
--batch-size 128 \
--device cuda:0 \
--preprocessing "ME" \
--dataset-size 12800 \
--skip-epochs 1450 \
--deterministic \
--save-model /data/lzj/easy/result/1027/res12_64_cifarfs_origin/res12_64_cifarfs_official.pkl

