# CUDA_VISIBLE_DEVICES=5 python main_word_embed_relations.py \
# --dataset-path /data1/lzj/easy_mine \
# --dataset miniimagenet \
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
# --save-model ./result/ablation/20240303_res12_miniimagenet_word_embed_relation/0.6relationloss_only.pkl \
# --attriloss_weight 0 \
# --relationloss_weight 0.6 \
# --attri-bank ./word_embedding/miniimagenet/train_vocab_vec.pkl \
# --class-relation ./word_embedding/miniimagenet/train_vec_simi.pkl

CUDA_VISIBLE_DEVICES=5 python main_word_embed_relations.py \
--dataset-path /data1/lzj/easy_mine \
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
--save-model ./result/ablation/20240303_res12_cifarfs_word_embed_relation/0.5attriloss_0.3relationloss.pkl \
--attriloss_weight 0.5 \
--relationloss_weight 0.3 \
--attri-bank ./word_embedding/cifarfs/train_vocab_vec.pkl \
--class-relation ./word_embedding/cifarfs/train_vec_simi.pkl

# CUDA_VISIBLE_DEVICES=3 python main_word_embed_relations.py \
# --dataset-path /data1/lzj/easy_mine \
# --dataset miniimagenet \
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
# --save-model ./result/ablation/20240303_res12_miniimagenet_word_embed_relation/0.6attriloss_0.6relationloss.pkl \
# --attriloss_weight 0.6 \
# --relationloss_weight 0.6 \
# --attri-bank ./word_embedding/miniimagenet/train_vocab_vec.pkl \
# --class-relation ./word_embedding/miniimagenet/train_vec_simi.pkl