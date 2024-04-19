##### train mini-ImageNet ######
# CUDA_VISIBLE_DEVICES=0 python main_word_embed.py \
# --dataset-path /data/lzj/easy \
# --dataset miniimagenet \
# --model resnet12 \
# --epochs 0 \
# --manifold-mixup 500 \
# --rotations \
# --cosine \
# --gamma 0.9 \
# --lr 0.1 \
# --milestones 100 \
# --batch-size 128 \
# --preprocessing ME \
# --n-shots [1,5] \
# --skip-epochs 480 \
# --save-model /data/lzj/easy/result/1027/res12_64_word_embed/resnet12_word64_embed_MSEME.pkl

##### train cifarfs ######
# CUDA_VISIBLE_DEVICES=0 python main_word_embed.py \
# --dataset-path /data/lzj/easy \
# --dataset cifarfs \
# --model resnet12 \
# --milestones 300 \
# --epochs 0 \
# --manifold-mixup 1500 \
# --cosine \
# --gamma 0.9 \
# --rotations \
# --batch-size 128 \
# --device cuda:0 \
# --preprocessing "ME" \
# --dataset-size 12800 \
# --skip-epochs 1450 \
# --deterministic \
# --save-model /data/lzj/easy/result/1027/res12_64_cifarfs_word_embed/resnet12_word64_embed_MSEME.pkl

##### train tiered-imagenet ######
# CUDA_VISIBLE_DEVICES=0 python main_word_embed.py \
# --dataset-path /data/lzj/easy \
# --dataset tieredimagenet \
# --model resnet12 \
# --milestones 300 \
# --epochs 0 \
# --manifold-mixup 300 \
# --load-model /data/lzj/easy/result/1027/res12_351_tiered_word_embed/resnet12_word351_embed_MSEME.pkl \
# --cosine \
# --gamma 0.9 \
# --lr 0.0729 \
# --rotations \
# --batch-size 128 \
# --device cuda:0 \
# --preprocessing "ME" \
# --dataset-size 12800 \
# --skip-epochs 280 \
# --deterministic \
# --save-model /data/lzj/easy/result/1027/res12_351_tiered_word_embed/resnet12_word351_embed_MSEME_continue.pkl

##### train cubfs ######
CUDA_VISIBLE_DEVICES=3 python main_word_embed.py \
--dataset-path /data/lzj/easy \
--dataset cubfs \
--model resnet12 \
--milestones 100 \
--epochs 0 \
--manifold-mixup 600 \
--cosine \
--gamma 0.9 \
--rotations \
--batch-size 128 \
--device cuda:0 \
--preprocessing "ME" \
--skip-epochs 580 \
--save-model /data/lzj/easy/result/1027/res12_cubfs_word_embed/resnet12_cubfs_word_embed.pkl