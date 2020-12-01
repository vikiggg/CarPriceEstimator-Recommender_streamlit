import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

PATH='data/'

label=pickle.load(open(PATH+'label_encoder.txt','rb'))
trained_model=pickle.load(open(PATH+'trained_model.txt','rb'))
# ========= Data: import ============
list_of_make=pickle.load(open(PATH+'list_of_make.txt','rb'))
default_model=pickle.load(open(PATH+'models_of_make_dict.txt','rb'))
default_drive_dict=pickle.load(open(PATH+'default_drive_by_car.txt','rb'))
default_cylinders_dict=pickle.load(open(PATH+'default_cylinders_by_car.txt','rb'))
default_fuel_dict=pickle.load(open(PATH+'default_fuel_by_car.txt','rb'))
default_transmission_dict=pickle.load(open(PATH+'default_transmission_by_car.txt','rb'))
default_size_dict=pickle.load(open(PATH+'default_size_by_car.txt','rb'))
default_type_dict=pickle.load(open(PATH+'default_type_by_car.txt','rb'))
title_status_list=pickle.load(open(PATH+'list_title_status.txt','rb'))
paint_color_list=pickle.load(open(PATH+'list_paint_color.txt','rb'))

rec_1=pd.read_pickle(PATH+'rec_data_1.zip',compression='zip')
rec_2=pd.read_pickle(PATH+'rec_data_1.zip',compression='zip')

def get_rec_data(rec_1,rec_2):
    rec=rec_1.merge(rec_2)
    return rec

# ========= Recommendation =========
rec_features=['year', 'manufacturer',
     'model', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status',
     'transmission', 'drive', 'size', 'type', 'paint_color','price']
res_features=['description','price', 'year', 'manufacturer','model', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status','transmission','drive', 'size', 'type', 'paint_color','state']
label_features=['year','manufacturer','model','condition','cylinders', 'fuel','title_status','transmission','drive','size','type','paint_color']

def upper(text):
     return text.upper()

