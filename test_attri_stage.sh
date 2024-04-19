
# CUDA_VISIBLE_DEVICES=2 python main_train_attri_stage.py \
# --dataset-path /data/lzj/easy \
# --dataset miniimagenet \
# --model resnet12 \
# --epochs 0 \
# --load-model /data/lzj/easy/result/0818/mini1.pt1 \
# --load-attri-model /data/lzj/easy/result/0911/res12frozen_attri/attri_model_1.pt1 \
# --memorybank-path /data/lzj/easy/save_tensor/raw_feat/feature_base80_res12/proto_feat_sort_stage10_1.pkl \
# --save-features /data/lzj/easy/save_features/0919/res12only_attri_stage5_cnt/feature_test.pt \
# --n-shots 1 \
# --batch-size 128 \
# --sample-aug 30 \
# --preprocessing ME 


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
# CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
# --dataset-path /data/lzj/easy \
# --dataset miniimagenet \
# --model resnet12 \
# --load-attri-model /data/lzj/easy/result/0908/res12frozen_attri_cos/attri_model_1.pt1 \
# --memorybank-path /data/lzj/easy/save_tensor/raw_feat/proto_attri_stage5_cnt_1.pkl \
# --test-features "/data/lzj/easy/save_features/0818/res12only/minifeaturesAS_2.pt1" \
# --preprocessing ME \
# --batch-fs 1 \
# --n-shots 1

CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
--dataset-path /data/lzj/easy \
--dataset miniimagenet \
--model resnet12 \
--load-attri-model /data/lzj/easy/result/0908/res12frozen_attri_cos/attri_model_1.pt1 \
--memorybank-path /data/lzj/easy/save_tensor/raw_feat/proto_feat_M_sort_1.pkl \
--test-features "/data/lzj/easy/save_features/0919/res12only_attri_stage5_cnt/feature_test.pt1" \
--preprocessing ME \
--batch-fs 1 \
--n-shots 1

# ''

# /data/lzj/easy/save_features/0831/minifeaturesAS_attri_1.pt1,/data/lzj/easy/save_features/0831/minifeaturesAS_attri_2.pt1,/data/lzj/easy/save_features/0831/minifeaturesAS_attri_3.pt1 