from tkinter import simpledialog
import gensim
import numpy as np
import torch
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# input_file = '/data/lzj/glove.6B.300d.txt'
# output_file = '/data/lzj/glove_2_word2vec.840B.300d.txt'

# glove2word2vec(input_file, output_file)
# print('done')
# model = KeyedVectors.load_word2vec_format(output_file)

model = KeyedVectors.load_word2vec_format('/data/lzj/GoogleNews-vectors-negative300.bin', binary=True)
# model.wv.save_word2vec_format('data/lzj/GoogleNews-vectors-negative300.bin')
# vocab = model.index2word
# f = open('/data/lzj/easy/vocab_glove840B.txt','a')
# print(type(vocab))
# for i in range(len(vocab)):
#     # print(i)
#     if i % 10000 == 0:
#         print(i)
#         print(vocab[i])
#     # temp = vocab[i]
#     f.write(vocab[i])
#     f.write("\n")
# f.close()
# word_distance = model.distance("robin", "lion", "dog") 'saluki', 'adult_Tibetan_mastiffs'

# print(model.similar_by_word('eyeball', 20))

# print(model.similarity('dog', 'boxers'))
# a = model['English_mastiff']
# b = model['dog']
# simi = np.dot(a/np.linalg.norm(a, ord=2), b/np.linalg.norm(b, ord=2))
# print(simi)




train_vocab = ['Brambling','goldfinch','finches','snowbird','indigo_bunting','robins','bulbul','blue_jay','magpie','chickadees','dipper',
        'striped_gecko','iguana','chameleon','whiptail','anole_lizards','bearded_dragon_lizard','alligator_lizard','venomous_lizards','basilisk_lizard',
        'veiled_chameleon','Komodo_dragon','slithery_snakes','poisonous_snake','hognose_snake','rat_snake','kingsnake','garter_snake',
        'constrictor_snake','indigo_snake','cottonmouth_snake','boa_constrictor','reticulated_python','venomous_cobra','green_mamba',
        'sea_serpent','vipers','diamondback_rattlesnake','rattlesnake','drakes','red_breasted_mergansers','goose','swan',
        'wood_stork','painted_storks','spoonbill','flamingo','blue_heron','egret','bittern','Sarus_crane','corncrakes','gallinule',
        'guinea_hen','bustards','ruddy_turnstones','dunlins','redshank','dowitchers','oystercatcher','pelican','emperor_penguins',
        'albatross','Afghan_hound','Basset_hound','beagles','bloodhound','bluetick_coonhound','tan_coonhound','coon_hound',
        'foxhounds','redbone_hound','Russian_wolfhound','Irish_wolfhound','greyhounds','whippet','Ibizan_hound','Norwegian_Elkhound',
        'giant_schnauzer','saluki','Scottish_deerhound','Weimaraners','staffordshire_terrier','American_Staffordshire_terriers',
        'Bedlington_terrier','Border_Terrier','Scottish_Terrier','haired_terriers','Norfolk_terrier','poodle_terrier','Yorkshire_terrier',
        'wire_fox_terrier','Airedale_Terrier','Sealyham_terrier','Airedale_terrier','Cairn_Terrier','haired_terriers','Dandie_Dinmont_terrier',
        'Boston_Terrier','miniature_schnauzer','giant_schnauzer','schnauzers','Scottish_terrier','Tibetan_Terrier','silky_terrier',
        'wheaten_terrier','Highland_terrier','Lhasa_apso','tabby_cat','ocelot','longhaired_cat','Siamese_cat','serval_cat','cougar',
        'lynx','leopard','snow_leopard','jaguar','lion','tiger','cheetah','sorrel','zebra','pig','wild_boar','warthog','hippopotamus',
        'ox','buffalo','bison','ram','bighorn_sheep','ibex','hartebeest','impala','gazelle','camel','llama','orangutan','gorilla',
        'chimpanzee','gibbons','siamang','guenon','patas_monkeys','baboon','macaque','langur','colobus_monkeys','proboscis_monkey','marmoset',
        'capuchin_monkey','howler_monkey','titi_monkey','spider_monkey','squirrel_monkey','ring_tailed_lemur','indri','abaya','piano_accordion',
        'acoustic_guitar','aircraft_carrier','airliner','airship','clock','apiary','assault_rifle','bakery','ballon','banjo','barbershop',
        'barn','barometer','baseball','basketball','bassoon','bell_tower','bikini','binder','binoculars','birdhouse','boathouse','bolo_tie',
        'bookshop','bows','bowties','brassiere','breastplate','buckle','butcher_shop','cannon','canoe','Bottle_Opener','cardigan',
        'Cassette_Player','catamaran','Cd_Player','cello','cellular_telephones','chain_maille','chimes','church','Movie_Theaters','cleaver',
        'locks','Computer_Keyboards','confectionery','Container_Ship','corkscrew','cornet','croquet','armor','rotary_dial_telephone','diaper',
        'countdown_clocks','Casio_watches','disc_brakes','dome','drum','electric_guitar','feather_boa','fireboat','fireguard','flute',
        'french_horn','fur_coat','gasmask','golfball','gondola','gong','grand_piano','greenhouse','grocery_store','guillotine','hairpin',
        'hammer','harmonica','harp','hatchet','holster','HOME_THEATER','hoopskirts','hourglass','iPod','blue_jeans','T_shirt','jigsaw_puzzle',
        'kimono','knots','lab_coats','lampshade','lawn_mower','covers','penknife','library','lifeboat','ocean_liner','loupe','magnetic_compass',
        'maillot','maraca','marimba','miniskirt','missile','modem','monastery','monitor','mosque','mosquito_nets','muzzle','nails','oboe',
        'ocarina','odometer','pipe_organ','oscilloscope','overskirt','padlock','palace','panpipe','Parking_Meter','payphone','helmet',
        'picket_fence','pirate_ship','pinwheel','planetarium','plow','plunger','poncho','billiard_table','projectile','projector','punching_bag',
        'radiotelescope','restaurant','revolver','rifle','rugby','ruler','bobby_pins','sarong','saxophone','scabbard','weighing_scales',
        'schooner','screw','screwdriver','seatbelt','shield','Shoe_Shop','shovel','ski_mask','sleeping_bag','space_shuttle','speedboat',
        'Steel_Drum','stethoscope','stole','stopwatch','stupa','submarine','suit','sundial','sunglasses','sweatshirt','swimming_trunks',
        'syringe','tape_recorder','tennis','thatched_roof','thimble','tile_roofs','tobacconist_shop','toyshop','trench_coat','trimaran',
        'trombone','umbrella','upright_piano','vault','violin','volleyball','antique_clocks','warplane','window','window_blinds','tie',
        'shipwreck','yawl','crossword_puzzle']
            
test_vocab = ['tench','goldfish','megalodon','tiger_shark','hammerhead_shark',
        'thornback_rays','stingray','Great_Pyrenees','schipperke','Belgian_shepherd',
        'malinois','Briard','kelpie','Komondor','English_sheepdog',
        'Shetland_sheepdog','collie','border_collie','Bouvier_des_Flandres','Rottweiler',
        'German_shepherd','Doberman_pinscher','miniature_pinscher','Bernese_Mountain_Dog','Bernese_mountain_dog',
        'Bernese_Mountain_dog','Entlebucher_Mountain_Dog','English_mastiff','bull_mastiff','adult_Tibetan_mastiffs',
        'french_bulldog','Great_Dane','Saint_Bernards','husky','malamute',
        'Siberian_husky','affenpinscher','tiger_beetle','ladybugs','blister_beetle',
        'long_horned_beetle','leaf_beetle','dung_beetle','rhinoceros_beetle','weevil',
        'flys','bee','ant','grasshopper','crickets',
        'Stick_Insect','cockroach','mantis','cicada','leafhopper',
        'lacewing','dragonfly','damselfly','Papilio',
        'Purple_Butterfly','monarch_butterfly','hairstreak_butterfly','Quino_checkerspot_butterfly','Morpho_butterfly',
        'snoek','eel','coho','butterflyfish','clownfish',
        'sturgeon','garfish','lionfish','pufferfish','bannister',
        'barrels','bathtub','beaker','beer_bottles','breakwater',
        'bucket','caldron','chainlink_fence','coffee_mug','coffeepot',
        'dam','Pressed_Powder','grille','ladle','mortar',
        'picket_fence','pill_bottles','pitcher','soda_bottle','Rain_Barrel',
        'Sliding_Door','drystone_wall','teapot','tub','turnstile',
        'vase','washbasin','waterbottle','jug','Watertower',
        'antique_jugs','Wine_Bottle','danged_fence','menu','plate',
        'guacamole','consomme','hotpot','sherry_trifle','icecream',
        'ice_lolly','cheeseburger','hotdog','cabbage','broccoli',
        'cauliflower','zucchini','spaghetti_squash','acorn_squash','butternut_squash',
        'cucumber','artichoke','bell_pepper','cardoon','mushroom',
        'Granny_Smith','strawberry','orange','lemon','figs',
        'pineapple','banana','jackfruit','custard_apple','pomegranate',
        'hay','carbonara','chocolate_sauce','dough','pizza',
        'potpie','burrito','vino','espresso','cup',
        'eggnog','alp','cliff','coral_reef','geyser',
        'lakeside','promontory','sandbar','seashore','valley',
        'volcano'
]


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
torch.save(train_vocab_vec, '/data/lzj/easy_mine/word_embedding/tiered/train_vocab_vec.pkl')
torch.save(test_vocab_vec, '/data/lzj/easy_mine/word_embedding/tiered/test_vocab_vec.pkl')





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