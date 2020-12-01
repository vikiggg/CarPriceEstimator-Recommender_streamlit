# Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import datetime
import re
import pickle
import string


#PATH='/content/drive/MyDrive/LHL/final/data/'
PATH='data/'

#====================Load dictionaries====================
age_meter=pickle.load(open(PATH+'age_meter_dict','rb'))

default_drive_dict=pickle.load(open(PATH+'default_drive.txt','rb'))
default_cylinders_dict=pickle.load(open(PATH+'default_cylinders.txt','rb'))
default_fuel_dict=pickle.load(open(PATH+'default_fuel.txt','rb'))
default_transmission_dict=pickle.load(open(PATH+'default_transmission.txt','rb'))
default_size_dict=pickle.load(open(PATH+'default_size.txt','rb'))
default_type_dict=pickle.load(open(PATH+'default_type.txt','rb'))
default_paint_color_dict=pickle.load(open(PATH+'default_paint_color.txt','rb'))
default_title_status_dict=pickle.load(open(PATH+'default_title_status.txt','rb'))
cat_features=['manufacturer',
       'model', 'condition', 'cylinders', 'fuel', 'title_status',
       'transmission', 'drive', 'size', 'type', 'paint_color']

# make --> default model
default_model_dict=pickle.load(open(PATH+'default_model_by_make.txt','rb'))
mode_dict=pickle.load(open(PATH+'mode_cat_features.txt','rb'))
#====================Functions====================
class Transformer():
    def __init__(self, func):
        self.func = func

    def fit_transform(self,X,*y):
        return self.func(X)

    def transform(self, X,*y, **transform_params):
        return self.func(X)

    def fit(self, X, *y, **fit_params):
        return self

def make_lower(text):
    return text.lower()

def outlier_age_meters(X):
    X['age']=X['year'].apply(car_age)
    X['odometer']=X['odometer'].apply(meters_outlier)
    X['std_meters']=X['age'].apply(car_st_meter)
    X['odometer'].fillna(X['std_meters'],inplace=True)
    X.drop('std_meters',axis=1,inplace=True)
    return X

def fill_default_model(df):
  lst=[]
  for car in df['manufacturer']:
    if car==np.nan:
      lst.append(default_model_dict[car])
    else:
      lst.append('')
  df['default_model']=lst
  df['model'][df['model']=='nan']=np.nan
  df['model'][df['model']=='NAN']=np.nan
  df['model'].fillna(df['default_model'],inplace=True)
  return df

def revise_manu(cars):
    ind=[] 
    manufacturer=[]
    drops=[]
    for idx in cars[cars['manufacturer'].isnull()].index:
        cars['model'][idx]=' '.join(re.sub(r'[19,20]\d+','',cars['model'][idx]).split())
        if re.search(r'[Ff]\W?\d+',cars['model'][idx]) or re.findall(r'superduty|mstang|ford',cars['model'][idx].lower()): 
            ind.append(idx)
            manufacturer.append('ford')
            #model.append(re.findall(r'[Ff]\W?\d+',cars['model'][idx]))
        elif re.findall(r'maserati',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('maserati')
        elif re.findall(r'corvette|cheverolet|camaro|chverolet|chryler|cheverolt',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('cheverolet')
        elif re.findall(r'acterra|sterling|sterling acterra',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('sterling acterra')
        elif re.findall(r'cruiser|chrysler|chryler',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('chrysler')
        elif re.findall(r'volkswagon|volkwagen|wagen|passet|jetta|tiguan',cars['model'][idx].lower()):
            ind.append(idx) 
            manufacturer.append('volkswagen')
        elif re.findall(r'suzuki',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('suzuki')
        elif re.findall(r'porsche|speedster|boxster|caddy|cts|turbo',cars['model'][idx].lower()) :
            ind.append(idx)
            manufacturer.append('porsche')
        elif re.findall(r'mini|cooper|bmw|mwb|bwm|x5|x3|x1|335|320',cars['model'][idx].lower()) :
            ind.append(idx)
            manufacturer.append('bmw')
        elif re.findall(r'international|freightliner|truck|kaiser|cascadia',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('truck')
        elif re.findall(r'mazda',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('mazda')
        elif re.findall(r'infinite|infiniti',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('infiniti')
        elif re.findall(r'mclaren',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('mclaren')
        elif re.findall(r'rollsroyce',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('rollsroyce')
        elif re.findall(r'benz|mercedes|smart|mb|ml\d+|cls\d+|sls\d+|sl\d+|cla\d+',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('mercedes-benz')
        elif re.findall(r'huyndai|sonata|hyndai',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('hyundai')
        elif re.findall(r'isuzu',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('isuzu')
        elif re.findall(r'bentley|bently',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('bentley')
        elif re.findall(r'lamborghini',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('lamborghini')
        elif re.findall(r'jeep',cars['model'][idx].lower()) or re.findall(r'hummer',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('jeep')
        elif re.findall(r'scion',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('scion')
        elif re.findall(r'accord|honda',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('honda')
        elif re.findall(r'nissian|rouge|rogue|altima',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('nissan')
        elif re.findall(r'toyota|tacoma|corolla|toyo\W|camry',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('toyota')
        elif re.findall(r'caravan|journey',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('dodge')
        elif re.findall(r'triumph|spitfire|ply|plymouth|spartan',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('others')
        elif re.findall(r'encore|buick',cars['model'][idx].lower()):
            ind.append(idx)
            manufacturer.append('buick')
        elif re.findall(r'nan|any|all',cars['model'][idx].lower()):
            cars.drop(idx,axis=0,inplace=True)
        try:
            if re.findall(r'truck|pickup',cars['type'][idx].lower()):
                ind.append(idx)
                manufacturer.append('truck')
        except:
            pass
        else:
            drops.append(idx)
    for i,idx in enumerate(ind):
        cars['manufacturer'][idx]=manufacturer[i]
    cars.drop(drops,axis=0,inplace=True)
    print (f'revised {len(ind)} car\'s manufacturers.')
    print (f'droped {len(drops)} cars.')
    cars=cars[~cars['manufacturer'].isnull()]
    print (f'left {cars.shape[0]} cars.')
    return cars

# revise manufacturers
def revise_manufacturers(cars):
  cars[cars['manufacturer']=='CHEVEROLET']['manufacturer']='CHEVROLET'
  cars[cars['manufacturer']=='HUYNDAI']['manufacturer']='HYUNDAI'
  cars[cars['manufacturer']=='NISSIAN']['manufacturer']='NISSAN'
  cars[cars['manufacturer']=='PORCHE']['manufacturer']='PORSCHE'
  cars[cars['manufacturer']=='VOLKSWAGON']['manufacturer']='VOLKSWAGEN'
  return cars

def fill_na_by_default(df):
    #drive
    df['default_drive']=[default_drive_dict[car] for car in df['car_model']]
    #cylinders
    df['default_cylinders']=[default_cylinders_dict[car] for car in df['car_model']]
    #fuel
    df['default_fuel']=[default_fuel_dict[car] for car in df['car_model']]
    #transmission
    df['default_transmission']=[default_transmission_dict[car] for car in df['car_model']]
    # size
    df['default_size']=[default_size_dict[car] for car in df['car_model']]
    # type
    df['default_type']=[default_type_dict[car] for car in df['car_model']]
    # paint_color
    df['default_paint_color']=[default_paint_color_dict[car] for car in df['car_model']]
    # title_status
    df['default_title_status']=[default_title_status_dict[car] for car in df['car_model']]

    df['drive'].fillna(df['default_drive'],inplace=True)
    df['cylinders'].fillna(df['default_cylinders'],inplace=True)
    df['fuel'].fillna(df['default_fuel'],inplace=True)
    ['car_model']
    df['transmission'].fillna(df['default_transmission'],inplace=True)
    df['size'].fillna(df['default_size'],inplace=True)
    df['type'].fillna(df['default_type'],inplace=True)
    df['paint_color'].fillna(df['default_paint_color'],inplace=True)
    df['title_status'].fillna(df['default_title_status'],inplace=True)
    return df

def fill_freq(cars):
    for col in ['condition', 'cylinders', 'fuel','odometer', 'title_status', 'transmission', 'drive', 'size', 'type',
       'paint_color']:
        cars[col].fillna(mode_dict[col],inplace=True)
    return cars

# =========== main functioon ========
def fill_nulls(cars):
    cars=cars[~cars['year'].isnull()]
    cars=cars[~((cars['manufacturer'].isnull())&(cars['model'].isnull()))]
    print('Removed rows lack of information')
    cars=cars[~(cars['price']==0)]
    print ('Removed irreasonable rows which price==0')
    cars=Transformer(outlier_age_meters).fit_transform(cars)
    print('Filled odometer')
    cars=fill_default_model(cars)
    print ('Filled default models')
    cars['model']=cars['model'].astype(str).apply(make_lower)
    cars=revise_manu(cars)
    cars['manufacturer']=cars['manufacturer'].apply(rm_punctuation)
    cars['model']=cars['model'].apply(rm_punctuation)
    cars=revise_manufacturers(cars)
    print ('Filled manufacturers')
    cars['car_model'] = list(zip(cars.manufacturer, cars.model))
    print ('Added a column (manufacturer,model)')
    cars=fill_na_by_default(cars)
    print ('Filled partial na by default values of the certain car model')
    cars=fill_freq(cars)
    print ('Filled all nulls')
    cars=cars[['year', 'manufacturer',
       'model', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status',
       'transmission', 'drive', 'size', 'type', 'paint_color','price']]
    return cars

# ============ Support Functions ========================
def rm_punctuation(text):
    return ''.join([w.upper() for w in text if w not in string.punctuation])

def normalized(X):
    X['odometer']=X['odometer'].apply(normal)
    return X

def normal(meter):
    res=meter/500000
    return res

def meters_outlier(odometer):
    if odometer>500000:
        odometer=500000
    if odometer==0:
        odometer==np.nan
    return odometer

def car_age(year):
    now_year=datetime.datetime.now().year
    age=now_year-year
    if age==np.nan:
        age=age
    else:
        age=round(age,0)
    if age<=0:
        age=0
    if age>=70:
        age=70
    return age

def car_st_meter(age):
    meter=age_meter[age]
    return meter

# ============ Modeling ========================
## encoder and normalization
label=pickle.load(open(PATH+'label_encoder.txt','rb'))
label_features=['year','manufacturer','model','condition','cylinders', 'fuel','title_status','transmission','drive','size','type','paint_color']

def ready_to_predict(df):
    input_df=df.copy()
    input_df[label_features]=label.transform(input_df[label_features])
    Transformer(normalized).transform(input_df)
    input_df[['year', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel',
       'title_status', 'transmission', 'drive', 'size', 'type',
       'paint_color']]=input_df[['year', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel',
       'title_status', 'transmission', 'drive', 'size', 'type',
       'paint_color']].astype('int')
    return input_df

