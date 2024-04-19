#### mini-ImageNet ####
# CUDA_VISIBLE_DEVICES=0 python save_memorybank.py \
# --dataset-path /data1/lzj/easy_mine \
# --dataset miniimagenet \
# --load-model ./result/ablation/20240303_res12_miniimagenet_word_embed_relation/0.6relationloss_only_67.91.pkl1 \
# --model resnet12 \
# --save-memorybank-path ./save_tensor/raw_feat/1120/new_feature_mini_base64_res12_0.6relationloss_only \
# --batch-size 64 

#### cifarfs ####
CUDA_VISIBLE_DEVICES=0 python save_memorybank.py \
--dataset-path /data1/lzj/easy_mine_new \
--dataset cifarfs \
--load-model ./result/ablation/20240303_res12_cifarfs_word_embed_relation/0.5attriloss_0.5relationloss_73.93.pkl1 \
--model resnet12 \
--save-memorybank-path ./save_tensor/raw_feat/1120/new_feature_cifarfs_base64_res12_0.5attriloss_0.5relationloss \
--batch-size 64 


# CUDA_VISIBLE_DEVICES=0 python save_memorybank.py \
# --dataset-path /data/lzj/easy \
# --dataset cifarfs \
# --load-model /data/lzj/easy/result/1027/res12_64_cifarfs_word_embed/resnet12_word64_embed_MSEME.pkl1 \
# --model resnet12 \
# --manifold-mixup 0 \
# --batch-size 128 \
# --preprocessing ME 
# --load-model /data/lzj/easy/result/1012/res12_64_word_embed/resnet12_word64_embed_MSEME_backfeat_continue.pkl1 \
# --load-attri-model /data/lzj/easy/result/0911/res12frozen_attri/attri_model_1.pt1 \
# --rotations \
# --cosine \

#### tieredimagenet ####
# CUDA_VISIBLE_DEVICES=0 python save_memorybank.py \
# --dataset-path /data/lzj/easy \
# --dataset cifarfs \
# --load-model /data/lzj/easy/result/1106/res12_64_cifarfs_word_embed_relation_clip/resnet12_word64_embed_relation_clip.pkl1 \
# --model resnet12 \
# --manifold-mixup 0 \
# --batch-size 128 \
# --preprocessing ME 