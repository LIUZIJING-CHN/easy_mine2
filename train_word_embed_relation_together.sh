##### train mini-ImageNet ######
# CUDA_VISIBLE_DEVICES=0 python main_word_embed_relations_together.py \
# --dataset-path /data/lzj/easy_mine \
# --dataset cifarfs \
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
# --save-model /data/lzj/easy_mine/result/1120/res12_64_cifar_word_embed_relation_together/resnet12_word64_embed_relation_together.pkl

# ##### train cifarfs ######
CUDA_VISIBLE_DEVICES=0 python main_word_embed_relations_together.py \
--dataset-path /data/lzj/easy_mine \
--dataset cifarfs \
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
--save-model /data/lzj/easy_mine/result/1120/res12_64_cifar_word_embed_relation_together/resnet12_word64_embed_relation_together.pkl

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
