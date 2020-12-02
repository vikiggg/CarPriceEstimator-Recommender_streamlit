import streamlit as st
from PIL import Image
from processing import *
from recommendation import *
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity

image = Image.open('car.jpg')
st.image(image,use_column_width=True)
st.write("""
# Car Price Estimator 
#### Welcome! :racing_car:
""")
st.info('Enter all required (Expecting)features in the sidebar!')
st.write('--'*30)

label=pickle.load(open(PATH+'label_encoder.txt','rb'))
trained_model=pickle.load(open(PATH+'trained_model.txt','rb'))
rec_data=get_rec_data(rec_1,rec_2)

# ========= Sidebar: UserInput ============
st.sidebar.title('Car Features')
make=st.sidebar.selectbox("Manufacturer",
list_of_make)
model=st.sidebar.selectbox("Car Model",default_model[make])
year=st.sidebar.slider('Year of make',1950,2020,2015)
odometer=st.sidebar.slider('Odometer(Mile)',0,500000,10000)
paint_color=st.sidebar.selectbox("Paint Color",paint_color_list)

drive=st.sidebar.selectbox("Drive",
default_drive_dict[(make,model)])
cartype=st.sidebar.selectbox("Car Type",
default_type_dict[(make,model)])
size=st.sidebar.selectbox("Size",
default_size_dict[(make,model)])
cylinders=st.sidebar.selectbox("Cylinders",default_cylinders_dict[(make,model)])
fuel=st.sidebar.selectbox("Fuel",default_fuel_dict[(make,model)])
transmission=st.sidebar.selectbox("Transmission",
default_transmission_dict[(make,model)])

title_status=st.sidebar.selectbox("Title Status",
title_status_list)
condition=st.sidebar.selectbox("Condition",
['excellent', 'good', 'like new', 'fair', 'new', 'salvage'])
#submit=st.sidebar.button('submit')

# ========= function ============
def sell_features():
    data = {'year':year,
    'manufacturer':make,
    'model':model,
    'condition':condition,
    'cylinders':cylinders,
    'fuel':fuel,
    'odometer':odometer,
    'title_status':title_status,
    'transmission':transmission,
    'drive':drive,
    'size':size,
    'type':cartype,
    'paint_color':paint_color}
    features = pd.DataFrame(data, index=[0])
    return features

def buy_features():
    data = {'year':year,
    'manufacturer':make,
    'model':model,
    'condition':'excellent',
    'cylinders':cylinders,
    'fuel':fuel,
    'odometer':odometer,
    'title_status':'clean',
    'transmission':transmission,
    'drive':drive,
    'size':size,
    'type':cartype,
    'paint_color':paint_color}
    features = pd.DataFrame(data, index=[0])
    return features
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
# ========= show on main page ============
left_column, right_column = st.beta_columns(2)
sell_car = left_column.button('Sell my car')
buy_car=right_column.button("Buy a used car")
if sell_car: 
    df = sell_features()
    df_to_predict=df.copy()
    df_to_recomm=df.copy()
    df_to_predict=ready_to_predict(df_to_predict)
    sell_pred=(trained_model.predict(df_to_predict))
    st.write(' ')
    st.write('Please review your car\'s features here:')
    st.dataframe(df)
    st.title(f'Your car estimately worth {int(sell_pred[0])} USD.')
    st.write('')
    st.title('See more similar deals -->')
    top5=get_sell_recommendation(df_to_recomm,rec_data,sell_pred)
    st.balloons()
    try:
        top5=top5[res_features]
        for i in range(top5.shape[0]):
                left_, right_ = st.beta_columns(2)
                try:
                    right_.table(top5.iloc[i,1:])
                except:
                    right_.write('Not Avaliable')
                try:
                    left_.write(top5.loc[i]['description'][:1200])
                except:
                    left_.write('No description here')
                st.write('='*80)
    except:
        st.warning('No similar deals on going at this moment.')
 
if buy_car:
    df = buy_features()
    df_to_predict=df.copy()
    df_to_recomm=df.copy()
    df_to_predict=ready_to_predict(df_to_predict)
    buy_pred=trained_model.predict(df_to_predict)
    st.write(' ')
    st.write('Please review your car\'s features here:')
    st.dataframe(df)
    st.title (f'The Estimated Price is {int(buy_pred[0])+2000} USD.')
    st.write(' ')
    st.title('See Alternative cars -->')
    top5=get_buy_recommendation(df_to_recomm,rec_data,buy_pred)
    st.balloons()
    try:
        top5=top5[res_features]
        for i in range(top5.shape[0]):
                left_, right_ = st.beta_columns(2)
                try:
                    right_.table(top5.iloc[i,1:])
                except:
                    right_.write('Not Avaliable')
                try:
                    left_.write(top5.iloc[i,0][:1200])
                except:
                    left_.write('No description here')
                st.write('='*80)
    except:
        st.warning('No similar deals on going at this moment.')

st.write(' ')    
expander = st.beta_expander("FAQ")
expander.write("""
#### Q1: How should I use this price estimator?
Answer: Please enter all features required above and click either "Buy" or "Sell" 
button to get estimated price, also you can get some other cars that are selling 
in the market which similar to yours.
\
#### Q2: Where can I buy the cars?
Answer: Contact the owner directly or contact a dealer.
\
#### Q3: I want to sell my car, but there is no option fit on mine?
Answer: That means your car model or configuration is not that common, please contact us for futher assistance.
""")