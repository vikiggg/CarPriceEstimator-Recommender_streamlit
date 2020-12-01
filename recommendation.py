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
def get_buy_recommendation(user_input,rec_data,pred_price):
    cars=rec_data[rec_features]
    user_input['price']=pred_price
    user_input[label_features]=label.transform(user_input[label_features])
    user_input=user_input[rec_features]
    cars[label_features]=label.transform(cars[label_features])
    cosine_similarities=pd.DataFrame(cosine_similarity(user_input,cars))
    top5=cosine_similarities.loc[0].sort_values(ascending=False).head().index.to_list()
    res_df=rec_data.iloc[top5]
    
    try:
        res_df['state']=res_df['state'].apply(upper)
    except:
        res_df['state']='NA'
   
    res_df.reset_index(drop=True,inplace=True)
    return res_df


def get_sell_recommendation(input_df,rec_data,pred_price):
     user_input=input_df.copy()
     cars=rec_data[(rec_data['manufacturer']==user_input['manufacturer'][0])&(rec_data['model'].str.contains(user_input['model'][0]))][rec_features].reset_index()

     if cars.shape[0]==0:
          res_df=[]
     else:
          user_input['price']=pred_price
          user_input[label_features]=label.transform(user_input[label_features])
          user_input=user_input[rec_features]
          cars[label_features]=label.transform(cars[label_features])
          cosine_similarities=pd.DataFrame(cosine_similarity(user_input,cars.drop('index',axis=1)))
          top5=cosine_similarities.loc[0].sort_values(ascending=False).head().index.to_list()
          res_df=cars.iloc[top5]
          res_df=res_df.set_index('index')
          lst=res_df.index.to_list()
          res_df=rec_data.loc[lst][res_features]
          try:
               res_df['state']=res_df['state'].apply(upper)
          except:
               res_df['state']='NA'

          res_df.reset_index(drop=True,inplace=True)
     return res_df

def upper(text):
     return text.upper()

