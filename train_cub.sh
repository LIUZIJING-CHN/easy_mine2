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

CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset-path /data/lzj/easy \
--dataset cubfs \
--model resnet12 \
--epochs 0 \
--manifold-mixup 500 \
--rotations \
--cosine \
--gamma 0.9 \
--lr 0.1 \
--milestones 100 \
--batch-size 256 \
--preprocessing ME \
--n-shots [1,5] \
--skip-epochs 480 \
--save-model /data/lzj/easy/result/1027/res12_100_cub_origin/res12_cub_origin.pkl