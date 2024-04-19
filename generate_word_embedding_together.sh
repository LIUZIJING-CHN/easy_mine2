#### mini-ImageNet ####
CUDA_VISIBLE_DEVICES=0 python generate_word_embedding_together.py \
--dataset-path /data/lzj/easy_mine \
--dataset cifarfs \
--load-model /data/lzj/easy_mine/result/1120/res12_64_cifar_word_embed_relation_together/resnet12_word64_embed_relation_together.pkl1 \
--model resnet12 \
--manifold-mixup 0 \
--batch-size 128 \
--preprocessing ME 
