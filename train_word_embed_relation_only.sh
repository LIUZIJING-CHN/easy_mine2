##### train mini-ImageNet ######
CUDA_VISIBLE_DEVICES=0 python main_word_embed_relations_only.py \
--dataset-path /data/lzj/easy_mine \
--dataset miniimagenet \
--model resnet12 \
--epochs 0 \
--manifold-mixup 500 \
--rotations \
--cosine \
--gamma 0.9 \
--lr 0.1 \
--milestones 100 \
--batch-size 128 \
--preprocessing ME \
--n-shots [1,5] \
--skip-epochs 480 \
--save-model /data/lzj/easy_mine/result/1204/res12_mini_word_embed_relation_only/resnet12_word64_embed_relation_only.pkl

# ##### train cifarfs ######
# CUDA_VISIBLE_DEVICES=0 python main_word_embed_relations.py \
# --dataset-path /data/lzj/easy_mine \
# --dataset cifarfs \
# --model resnet12 \
# --milestones 20 \
# --epochs 0 \
# --manifold-mixup 20 \
# --load-model /data/lzj/easy_mine/result/1120/res12_64_cifarfs_word_embed_relation/resnet12_word64_embed_relation.pkl \
# --cosine \
# --gamma 0.9 \
# --lr 0.005 \
# --rotations \
# --batch-size 128 \
# --device cuda:0 \
# --preprocessing "ME" \
# --skip-epochs 0 \
# --save-model /data/lzj/easy_mine/result/1120/res12_64_cifarfs_word_embed_relation/resnet12_word64_embed_relation_continue.pkl

# ##### train FC100 ######
# CUDA_VISIBLE_DEVICES=0 python main_word_embed_relations.py \
# --dataset-path /data/lzj/easy_mine \
# --dataset fc100 \
# --model resnet12 \
# --milestones 100 \
# --epochs 0 \
# --manifold-mixup 600 \
# --cosine \
# --gamma 0.9 \
# --rotations \
# --batch-size 128 \
# --device cuda:0 \
# --preprocessing "ME" \
# --skip-epochs 580 \
# --save-model /data/lzj/easy_mine/result/1120/res12_60_fc100_word_embed_relation/resnet12_word60_embed_relation.pkl


# ##### train tieredimagenet ######
# CUDA_VISIBLE_DEVICES=0 python main_word_embed_relations.py \
# --dataset-path /data/lzj/easy_mine \
# --dataset tieredimagenet \
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
# --skip-epochs 1480 \
# --deterministic \
# --batch-fs 10 \
# --save-model /data/lzj/easy_mine/result/1120/res12_351_tier_word_embed_relation/resnet12_word351_embed_relation_new.pkl


##### train cubfs ######
# CUDA_VISIBLE_DEVICES=0 python main_word_embed_relations.py \
# --dataset-path /data/lzj/easy \
# --dataset cubfs \
# --model resnet12 \
# --milestones 100 \
# --epochs 0 \
# --manifold-mixup 600 \
# --cosine \
# --gamma 0.9 \
# --rotations \
# --batch-size 128 \
# --device cuda:0 \
# --preprocessing "ME" \
# --skip-epochs 580 \
# --save-model /data/lzj/easy/result/1106/res12_64_cubfs_word_embed_relation/resnet12_word100_embed_relation.pkl
