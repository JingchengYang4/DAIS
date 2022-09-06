import pickle
from detectron2.checkpoint.c2_model_loading import *

resnet = []

with open('/home/jingcheng/.torch/fvcore_cache/detectron2/ImageNetPretrained/MSRA/R-50.pkl', 'rb') as f:
    data = pickle.load(f)
    i = data
    k = convert_c2_detectron_names(data)
    print(type(k[0]))
    print(type(k[1]))
    weights = k[0]
    map = k[1]

with open('../models/bts_res50.pkl', 'rb') as f:
    data = pickle.load(f)
    k = data['model']
    print(weights.keys())
    for x in k:
        if x in weights:
            print(map[x])
            i[map[x]] = k[x]

with open('../models/bts_res50_mod.pkl', 'wb') as handle:
    pickle.dump(i, handle, protocol=pickle.HIGHEST_PROTOCOL)