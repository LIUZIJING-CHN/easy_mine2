### test mini-ImageNet ####
CUDA_VISIBLE_DEVICES=3 python main_word_embed_relations_together.py \
--dataset-path /data/lzj/easy_mine \
--dataset miniimagenet \
--model resnet12 \
--epochs 0 \
--load-model /data/lzj/easy_mine/result/1120/res12_64_mini_word_embed_relation_together/resnet12_word64_embed_relation_together.pkl1 \
--save-features /data/lzj/easy_mine/save_features/1120/res12_64_mini_word_embed_relation_together/resnet12_64_word_embed_relation_together_1shot_origin_aug30.pt \
--n-shots 1 \
--batch-size 128 \
--sample-aug 30 \
--preprocessing ME \
# --use-supp \

## test cifarfs ####
# CUDA_VISIBLE_DEVICES=0 python main_word_embed_relations_together.py \
# --dataset-path /data/lzj/easy_mine \
# --dataset cifarfs \
# --model resnet12 \
# --epochs 0 \
# --load-model /data/lzj/easy_mine/result/1120/res12_64_cifar_word_embed_relation_together/resnet12_word64_embed_relation_together.pkl1 \
# --save-features /data/lzj/easy_mine/save_features/1120/res12_64_cifar_word_embed_relation_together/resnet12_64_word_embed_relation_together_weighted_1shot_aug30.pt \
# --n-shots 1 \
# --batch-size 128 \
# --sample-aug 30 \
# --preprocessing ME \
# --use-supp 

## test cubfs ####
# CUDA_VISIBLE_DEVICES=0 python main_word_embed_relations.py \
# --dataset-path /data/lzj/easy \
# --dataset cubfs \
# --model resnet12 \
# --epochs 0 \
# --load-model /data/lzj/easy/result/1106/res12_100_cubfs_word_embed_relation/resnet12_word100_embed_relation.pkl1 \
# --save-features /data/lzj/easy/save_features/1106/res12_cubfs_100class/resnet12_100_word_embed_MSEME_relation_1shot_relation_aug1.pt \
# --n-shots 1 \
# --batch-size 128 \
# --sample-aug 1 \
# --preprocessing ME \
# --use-supp 

## test tieredimagenet ####
# CUDA_VISIBLE_DEVICES=0 python main_word_embed.py \
# --dataset-path /data/lzj/easy \
# --dataset tieredimagenet \
# --model resnet12 \
# --epochs 0 \
# --load-model /data/lzj/easy/result/1027/res12_351_tiered_word_embed/resnet12_word351_embed_MSEME.pkl1 \
# --save-features /data/lzj/easy/save_features/1027/res12_tier_351class/res12_tiered_word_embed_aug1.pt \
# --n-shots 1 \
# --batch-size 128 \
# --preprocessing ME \
# --sample-aug 1 

# --memorybank-path /data/lzj/easy/save_tensor/raw_feat/feature_word_embed_res12/resnet12_word_embed_cosME.pkl \
# --class-relation /data/lzj/easy/tmp/test_train_simi.pkl 
# --load-model /data/lzj/easy/result/1012/res12_64_word_embed/resnet12_word64_embed_MSEME_backfeat_continue.pkl1 \
# --save-features /data/lzj/easy/save_features/1018/res12_64class/resnet12_64_word_embed_backfeat_MSEME_aug30.pt \

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
# CUDA_VISIBLE_DEVICES=0 python main_word_embed.py \
# --dataset-path /data/lzj/easy \
# --dataset miniimagenet \
# --model resnet12 \
# --test-features "/data/lzj/easy/save_features/1027/res12_64class/resnet12_64_word_embed_MSEME_5shot_aug30.pt5" \
# --preprocessing ME \
# --n-shots 5