CUDA_VISIBLE_DEVICES=0 python main_res50.py \
--dataset-path /data1/lzj/easy_mine \
--dataset cifarfs \
--model resnet12 \
--epochs 0 \
--load-model /data1/lzj/easy_mine/result/2023-12-27/res50/cifarfs_1e-3.pt \
--load-attri-model /data1/lzj/easy_mine/result/0911/res12frozen_attri/attri_model_1.pt1 \
--memorybank-path /data1/lzj/easy_mine/save_tensor/raw_feat/feature_res50/proto_feat_res50_stage10_1.pkl \
--save-features cifarfs_1e-3_feature_ME.pt \
--n-shots 1 \
--batch-size 128 \
--sample-aug 30 \
--preprocessing ME \
--no-train

# CUDA_VISIBLE_DEVICES=0 python main_train_attri.py \
# --dataset-path /data/lzj/easy \
# --dataset miniimagenet \
# --model resnet12 \
# --epochs 0 \
# --load-model /data/lzj/easy/result/0818/mini2.pt1 \
# --memorybank-path /data/lzj/easy/save_features/save_memorybank_res12/res12frozen_5x5/memorybank_2.pkl \
# --save-features /data/lzj/easy/save_features/0904/res12only_attri/minifeaturesAS_attri_15switch2_2.pt \
# --n-shots 1 \
# --batch-size 128 \
# --sample-aug 30 \
# --preprocessing ME

# CUDA_VISIBLE_DEVICES=2 python main_train_attri.py \
# --dataset-path /data/lzj/easy \
# --dataset miniimagenet \
# --model resnet12 \
# --epochs 0 \
# --load-model /data/lzj/easy/result/0818/mini4.pt1 \
# --memorybank-path /data/lzj/easy/save_features/save_memorybank_res12/res12frozen_5x5/memorybank_3.pkl \
# --save-features /data/lzj/easy/save_features/0904/res12only_attri/minifeaturesAS_attri_15switch2_3.pt \
# --n-shots 1 \
# --batch-size 128 \
# --sample-aug 30 \
# --preprocessing ME

# CUDA_VISIBLE_DEVICES=0 python main_train_attri.py \
# --dataset-path "/data/lzj/easy" \
# --dataset miniimagenet \
# --model resnet12 \
# --epochs 200 \
# --episodic \
# --load-model /data/lzj/easy/result/0818/mini2.pt1 \
# --manifold-mixup 0 \
# --episodes-per-epoch 500 \
# --gamma 0.3 \
# --lr 0.01 \
# --milestones [100,150] \
# --skip-epochs 180 \
# --batch-size 80 \
# --preprocessing ME \
# --save-attri-model "/data/lzj/easy/result/0908/res12frozen_attri_cos/attri_model_2.pt" \
# --n-shot [1,5] 

# CUDA_VISIBLE_DEVICES=0 python main_train_attri.py \
# --dataset-path /data/lzj/easy \
# --dataset miniimagenet \
# --model resnet12 \
# --epochs 0 \
# --load-model /data/lzj/easy/result/0818/mini2.pt1 \
# --load-attri-model /data/lzj/easy/result/0829/attribute_model2.pt1 \
# --memorybank-path /data/lzj/easy/save_features/save_memorybank_res12/res12frozen/memorybank_2.pkl \
# --save-features /data/lzj/easy/save_features/0904/res12only_attri/minifeaturesAS_attri_5switch2_2.pt \
# --n-shots 1 \
# --batch-size 128 \
# --sample-aug 30 

# CUDA_VISIBLE_DEVICES=0 python main_train_attri.py \
# --dataset-path /data/lzj/easy \
# --dataset miniimagenet \
# --model resnet12 \
# --epochs 0 \
# --load-model /data/lzj/easy/result/0818/mini4.pt1 \
# --load-attri-model /data/lzj/easy/result/0829/attribute_model3.pt1 \
# --memorybank-path /data/lzj/easy/save_features/save_memorybank_res12/res12frozen/memorybank_3.pkl \
# --save-features /data/lzj/easy/save_features/0904/res12only_attri/minifeaturesAS_attri_5switch2_3.pt \
# --n-shots 1 \
# --batch-size 128 \
# --sample-aug 30 

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
# CUDA_VISIBLE_DEVICES=2 python main_res50.py \
# --dataset-path /data/lzj/easy \
# --dataset miniimagenet \
# --model resnet12 \
# --load-attri-model /data/lzj/easy/result/0908/res12frozen_attri_cos/attri_model_1.pt1 \
# --memorybank-path /data/lzj/easy/save_tensor/raw_feat/proto_feat_M_1.pkl \
# --test-features "/data/lzj/easy/save_features/0927/res50only/minifeaturesAS_stage_1.pt1,/data/lzj/easy/save_features/0927/res50only/minifeaturesAS_stage_2.pt1,/data/lzj/easy/save_features/0927/res50only/minifeaturesAS_stage_3.pt1" \
# --preprocessing ME \
# --n-shots 1 \
# --batch-fs 1

# ''

# /data/lzj/easy/save_features/0831/minifeaturesAS_attri_1.pt1,/data/lzj/easy/save_features/0831/minifeaturesAS_attri_2.pt1,/data/lzj/easy/save_features/0831/minifeaturesAS_attri_3.pt1 

# CUDA_VISIBLE_DEVICES=0 python main_res50.py \
# --dataset-path /data1/lzj/easy_mine \
# --dataset miniimagenet \
# --model resnet12 \
# --test-features "/data1/lzj/easy_mine/save_features/1106/res12_mini_64class/resnet12_64_word_embed_relation_1shot_origin_aug30.pt1" \
# --preprocessing E \
# --n-shots 5