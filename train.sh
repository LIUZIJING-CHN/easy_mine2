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
# --dataset-path /data/lzj/easy_mine \
# --dataset cifarfs \
# --model resnet12 \
# --epochs 0 \
# --manifold-mixup 600 \
# --rotations \
# --cosine \
# --gamma 0.9 \
# --milestones 100 \
# --batch-size 128 \
# --preprocessing ME \
# --n-shots [1,5] \
# --skip-epochs 580 \
# --save-model /data/lzj/easy_mine/result/1204/res12_cifar_origin/res12_cifar64_origin.pkl

CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset-path /data1/lzj/easy_mine \
--dataset tieredimagenet \
--model resnet12 \
--epochs 0 \
--manifold-mixup 1500 \
--rotations \
--cosine \
--gamma 0.9 \
--milestones 300 \
--batch-size 128 \
--dataset-size 12800 \
--preprocessing ME \
--n-shots [1,5] \
--skip-epochs 1480 \
--deterministic \
--device cuda:0 \
--save-model ./result/1222/res12_tiered_origin/res12_tiered_origin.pkl

# CUDA_VISIBLE_DEVICES=0 python main.py \
# --dataset-path /data/lzj/easy \
# --dataset cubfs \
# --model resnet12 \
# --epochs 0 \
# --manifold-mixup 600 \
# --rotations \
# --cosine \
# --gamma 0.9 \
# --lr 0.1 \
# --milestones 100 \
# --batch-size 256 \
# --preprocessing ME \
# --n-shots [1,5] \
# --skip-epochs 580 \
# --save-model /data/lzj/easy/result/1027/res12_cubfs_origin/res12_cubfs_origin.pkl