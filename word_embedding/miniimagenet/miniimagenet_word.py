from tkinter import simpledialog
import gensim
import numpy as np
import torch
from gensim.models import Word2Vec, KeyedVectors

model = gensim.models.KeyedVectors.load_word2vec_format('/data/lzj/GoogleNews-vectors-negative300.bin', binary=True)
# model = gensim.models.KeyedVectors.load_word2vec_format('/data/lzj/glove_2_word2vec.840B.300d.txt', binary=False)

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

# print(model.similar_by_word('pinetree', 20))
# print(model.similar_by_word('palmtree', 20))
# print(model.similar_by_word('Bernese_mountain_dogs', 20))
# print(model.similar_by_word('Bernese_Mountain_dog', 20))

# print(model.similarity('sherry_trifle', 'trifle'))
# print(model.similarity('berry_trifle', 'cake'))
# print(model.similarity('berry_trifle', 'trifle'))

# print(model.similarity('dog', 'boxers'))
# a = model['English_mastiff']
# b = model['dog']
# simi = np.dot(a/np.linalg.norm(a, ord=2), b/np.linalg.norm(b, ord=2))
# print(simi)



# # more_sentences = ['Gordon_setter','boxer_dog','butterflyfish_tricolor','pencil_box','theatre_curtain']
# train_vocab = ['finches','robins','triceratops',
#             'green_mamba','daddy_longlegs','toucan','jellyfish',
#             'dugongs','Basset_hound','saluki','Irish_Setter',
#             'English_sheepdog','English_mastiff','Tibetan_mastiff_pups','french_bulldog',
#             'Labradors_Newfoundlands','miniature_poodle','Arctic_foxes','ladybugs',
#             'toed_sloth','butterflyfish','aircraft_carrier','trash_cans',
#             'barrels','beer_bottles','carousels','chimes',
#             'clogs','cocktail_shaker','dishrag','dome',
#             'filing_cabinets','fireguard','frying_pan','hairpin',
#             'holster','lipstick','oboe','pipe_organ',
#             'parallel_bars','pencil_pouch','photocopier','prayer_rugs',
#             'reels','slot_machine','snorkel','solar_panels',
#             'spider_web','stage','tank','tile_roofs',
#             'tobacconist_shop','unicycle','upright_piano','wok',
#             'fence','yawl','signpost','consomme',
#             'hotdog','oranges','cliff','bolete','corn',
            
#             'goldfish','cock','loggerhead_turtle','dowitchers','hamster',
#             'ram','guenon','pig','airliner','ambulance','barbershop',
#             'basketball','bows','bucket','church','parachute']
#             # 'treefrog','striped_gecko','Nile_crocodile','reticulated_python','kangaroo',
#             # 'snail','Maine_lobster','tabby_cat','Blackbears','cockroaches',
#             # 'ballpoint_pen','barn','wheelbarrow','Bottle_Opener','wheels',
#             # 'Freightcar','golfball','stupa','volleyball','cheeseburger']
            
test_vocab = ['nematode','king_crab','golden_retriever','malamute',
            'dalmatian','striped_hyena','lion','ants',
            'ferret','bookshop','crate','breastplate',
            'electric_guitar','hourglass','mixing_bowl','schoolbus',
            'scoreboard','curtain','vase','sherry_trifle']


# train_vocab_vec = []
test_vocab_vec = []
# for _, word in enumerate(train_vocab):
#     print(word)
#     vec = torch.from_numpy(model[word]).unsqueeze(0)
#     train_vocab_vec.append(vec)
for _, word in enumerate(test_vocab):
    print(word)
    vec = torch.from_numpy(model[word]).unsqueeze(0)
    test_vocab_vec.append(vec)

# train_vocab_vec = torch.cat(train_vocab_vec, dim=0)
test_vocab_vec = torch.cat(test_vocab_vec, dim=0)
# torch.save(train_vocab_vec, '/data/lzj/easy/tmp/train80_vocab_vec.pkl')
torch.save(test_vocab_vec, '/data/lzj/easy_mine/word_embedding/miniimagenet/test_vocab_vec.pkl')





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