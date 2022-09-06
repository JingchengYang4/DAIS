import pickle

i = []

with open('/home/jingcheng/Downloads/model_final_b275ba.pkl', 'rb') as f:
    data = pickle.load(f)
    print(type(data['model']['backbone.bottom_up.res4.2.conv2.weight']))
    print(data['model']['backbone.bottom_up.res4.2.conv2.weight'].shape)
    #print(data['model']['backbone.bottom_up.res4.2.conv2.weight'])
    i = data['model']

print("----")

with open('../models/bts_res50.pkl', 'rb') as f:
    data = pickle.load(f)
    print(type(data['model']['backbone.bottom_up.res4.2.conv2.weight']))
    print(data['model']['backbone.bottom_up.res4.2.conv2.weight'].size())

    k = data['model']
    #for x in k:
        #print(x)
