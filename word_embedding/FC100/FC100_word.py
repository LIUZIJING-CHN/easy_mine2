from tkinter import simpledialog
import gensim
import numpy as np
import torch
from gensim.models import Word2Vec, KeyedVectors

model = gensim.models.KeyedVectors.load_word2vec_format('/data/lzj/GoogleNews-vectors-negative300.bin', binary=True)
# model.wv.save_word2vec_format('data/lzj/GoogleNews-vectors-negative300.bin')
# vocab = model.index_to_key
# f = open('/data/lzj/easy/vocab.txt','a')
# print(type(vocab))
# for i in vocab:
#     # print(i)
#     # if i % 10000 == 0:
#     #     print(i)
#     #     print(vocab.keys[i])
#     # temp = vocab[i]
#     f.write(i)
#     f.write("\n")
# f.close()
# word_distance = model.distance("robin", "lion", "dog") 'saluki', 'adult_Tibetan_mastiffs'

# print(model.similarity('cichlid', 'goldfish'))
# print(model.similarity('cichlid', 'fish'))
# print(model.similarity('bony_fishes', 'goldfish'))
# print(model.similar_by_word('cichlid', 20))
# print(model.similar_by_word('goldfish', 20))

# print(model.similarity('dog', 'boxers'))
# a = model['English_mastiff']
# b = model['dog']
# simi = np.dot(a/np.linalg.norm(a, ord=2), b/np.linalg.norm(b, ord=2))
# print(simi)




# train_vocab = ['train','skyscraper','turtle','raccoon','spider',
#                'orange','castle','keyboard','clock','pear',
#                'girl','seal','elephant','apple','ornamental_fish',
#                'bus','mushroom','possum','squirrel','chair',
#                'tank','plate','wolf','road','mouse',
#                'boy','shrew','couch','sunflower','tiger',
#                'caterpillar','lion','streetcar','lawn_mower','tulip',
#                'forest','dolphin','cockroach','bear','porcupine',
#                'bee','hamster','lobster','bowl','can',
#                'bottle','trout','snake','bridge','pine_tree',
#                'skunk','lizard','cup','kangaroo','oak_tree',
#                'dinosaur','rabbit','orchid','willow_tree','ray',
#                'palm_tree','mountain','house','cloud']

# test_vocab = ['baby','bed','bicycle','chimpanzee','fox',
#               'leopard','man','pickup_truck','plain','poppy',
#               'rocket','rose','snail','sweet_peppers','table',
#               'telephone','wardrobe','whale','woman','worm']

train_vocab = ['apple','ornamental_fish','bed','bicycle','bottles',
        'bowls','bridge','bus','cans','castle',
        'chair','clock','cloud','couch','crocodile',
        'cups','dinosaur','flatfish','forest','house',
        'keyboard','lamp','lawn_mower','lizard','maple_tree',
        'motorcycle','mountain','mushrooms','oak_tree','oranges',
        'orchids','palm_tree','pears','pickup_truck','pine_tree',
        'plain','plates','poppies','ray','road',
        'rocket','roses','sea','shark','skyscraper',
        'snake','streetcar','sunflowers','sweet_peppers','table',
        'tank','telephone','television','tractor','train',
        'trout','tulips','turtle','wardrobe','willow_tree']
            
test_vocab = ['baby','beaver','bee','beetle','boy',
        'butterfly','caterpillar','cockroach','dolphin','fox',
        'girl','man','otter','porcupine','possum',
        'raccoon','seal','skunk','whale','woman']


train_vocab_vec = []
test_vocab_vec = []
for _, word in enumerate(train_vocab):
    print(word)
    vec = torch.from_numpy(model[word]).unsqueeze(0)
    train_vocab_vec.append(vec)
for _, word in enumerate(test_vocab):
    print(word)
    vec = torch.from_numpy(model[word]).unsqueeze(0)
    test_vocab_vec.append(vec)

train_vocab_vec = torch.cat(train_vocab_vec, dim=0)
test_vocab_vec = torch.cat(test_vocab_vec, dim=0)
torch.save(train_vocab_vec, '/data/lzj/easy_mine/word_embedding/FC100/train_vocab_vec.pkl')
torch.save(test_vocab_vec, '/data/lzj/easy_mine/word_embedding/FC100/test_vocab_vec.pkl')





# # 再加载第三方预训练模型：
# model = Word2Vec(size=300, min_count=1, hs=1) 
# print('done')

# # 通过 intersect_word2vec_format()方法merge词向量：
# model.build_vocab(more_sentences) 	 
# print('done')
	 
# model.intersect_word2vec_format('/data/lzj/GoogleNews-vectors-negative300.bin', binary=True, unicode_errors='ignore')
# # binary=False, lockf=1.0 
# print('done')

# model.train(more_sentences, epochs=50, total_examples=model.corpus_count)

# model.save('/data/lzj/easy/tmp/mymodel.bin')
# model.wv.save_word2vec_format('/data/lzj/easy/tmp/mymodel_vec.dict')