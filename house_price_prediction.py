import pandas as pd
import numpy as np
data=pd.read_csv('Bengaluru_House_Data.csv')
data.drop(columns=['area_type', 'availability', 'society', 'balcony'], inplace=True)
data['location']=data['location'].fillna("Sarjapur Road")
data['size']= data['size'].fillna('2 BHK')
data['bath'] = data['bath'].fillna(data['bath'].median())
data['bhk']=data['size'].str.split().str.get(0).astype(int)
def convertRange(x):

    temp = x.split('-')
    if len(temp) == 2:
        return (float(temp[0])+ float(temp[1]))/2
    try:
        #in case of Nan values it will through exception
        return float(x)
    except:
        return None
data[ 'total_sqft']=data['total_sqft'].apply(convertRange)
data['price_per_sqft'] = data['price'] *100000/data['total_sqft']
data['location'] = data['location'].apply(lambda x: x.strip())
location_count= data['location'].value_counts()
location_count_less_10 = location_count [location_count<=10]
data['location']=data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)
data = data[((data['total_sqft']/data['bhk']) >= 300)]
def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    # group key-location, value : subdata
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std (subdf.price_per_sqft)
        gen_df = subdf[(subdf.price_per_sqft> (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output, gen_df],ignore_index =True)

    return df_output
data = remove_outliers_sqft (data)
def bhk_outlier_remover(df):
    e=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk , bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                e=np.append(e,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(e,axis='index')


data=bhk_outlier_remover(data)
data.drop(columns=['size','price_per_sqft'],inplace=True)
data.to_csv('Cleaned_data3.csv')
x=data.drop(columns=['price']) 
y=data['price']
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer 
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle
X_train, X_test,y_train,y_test= train_test_split(x, y, test_size=0.2, random_state=0)
column_trans=make_column_transformer ((OneHotEncoder (sparse=False), ['location']),remainder='passthrough')
scaler=StandardScaler()                                     
Ir=LinearRegression(normalize=True)
pipe = make_pipeline (column_trans, scaler, Ir)
pipe.fit(X_train,y_train)
y_pred_1r = pipe.predict(X_test)
lasso= Lasso()
pipe=make_pipeline (column_trans, scaler, lasso)
pipe.fit(X_train, y_train)
y_pred_lasso = pipe.predict(X_test)
#applying Ridge
ridge = Ridge()
pipe = make_pipeline (column_trans, scaler, ridge)
pipe.fit (X_train,y_train)
y_pred_ridge = pipe.predict(X_test)
pickle.dump(pipe, open('RidgeModel.pkl', 'wb'))
def predict_(location,bhk,bath,sqft):
    input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    op=pipe.predict(input)[0]
    return str(op)
#print(predict_("5th Phase JP Nagar",4,3,2000))

from tkinter import *
import pandas as pd
from tkinter import messagebox
file=pd.read_csv("Cleaned_data.csv")
root = Tk()
root.geometry("500x500")
root.wm_iconbitmap("my.ico")
root.title('House Price Predictor')
label_0 =Label(root,text="Require Details", width=20,font=("bold",20))
label_0.place(x=90,y=60)

#----------Area------------------
#file['total_sqft'] = file['total_sqft'].unique()
lis=list(set(file['total_sqft']))
label_1 =Label(root,text="Area(sq ft)", width=20,font=("bold",10))
label_1.place(x=80,y=130)
per_Sqft=StringVar()
entry_1=Entry(root,textvariable=per_Sqft)
entry_1.place(x=240,y=130)


#---------Bathroom---------------
label_2 =Label(root,text="No. of Bathrooms", width=20,font=("bold",10))
label_2.place(x=68,y=180)
lis=range(int(list(set(file['bath']))[1]),int(list(set(file['bath']))[-1])+1)
bathroom=StringVar()
droplist2=OptionMenu(root,bathroom, *lis)
droplist2.config(width=15)
bathroom.set('No. of Bathroom')
droplist2.place(x=240,y=180)

#------------BHK----------------------------
label_3 =Label(root,text="BHK", width=20,font=("bold",10))
label_3.place(x=70,y=230)
lis=list(set(file['bhk']))
bhk=StringVar()
droplist3=OptionMenu(root,bhk, *lis)
droplist3.config(width=15)
bhk.set('Select BHK')
droplist3.place(x=240,y=230)

#----------------Location-----------------
label_4=Label(root,text="Location",width=20,font=("bold",10))
label_4.place(x=70,y=280)
list_of_location=list(set(file['location']))
location=StringVar()
droplist4=OptionMenu(root,location, *list_of_location)
droplist4.config(width=15)
location.set('Select Location')
droplist4.place(x=240,y=280)

def predict():
    Area=per_Sqft.get()
    Bathroom=bathroom.get()
    BHK=bhk.get()
    Location=location.get()
    l=[]
    if Area=='' or Bathroom=='No. of Bathroom' or BHK=='Select BHK' or Location=='Select Location':
        if Area=='':
            l.append('Area')
        if Bathroom=='No. of Bathroom':
            l.append(Bathroom)
        if BHK=='Select BHK':
            l.append(BHK)
        if Location=='Select Location':
            l.append("Location")
        s=",".join(l)
        messagebox.showwarning("Warning",f"Select Atleat one value for {s}")
        
    else:
        if Area.isnumeric():
            entry_1.destroy()
            droplist2.destroy()
            droplist3.destroy()
            droplist4.destroy()

            root.title('House Price Predictor')
            label_0 =Label(root,text="Predicted Price of\nGiven Details", width=20,font=("bold",20))
            label_0.place(x=90,y=40)
        
            label_1 =Label(root,text="Area(sq ft) : ", width=20,font=("bold",13))
            label_1.place(x=80,y=130)
            label1 =Label(root,text=Area, width=20,font=("bold",13))
            label1.place(x=240,y=130)
        
            label_2 =Label(root,text="No. of Bathrooms : ", width=20,font=("bold",13))
            label_2.place(x=80,y=180)
            label2 =Label(root,text=Bathroom, width=20,font=("bold",13))
            label2.place(x=240,y=180)
        
            label_3 =Label(root,text="BHK : ", width=20,font=("bold",13))
            label_3.place(x=80,y=230)
            label3 =Label(root,text=BHK, width=20,font=("bold",13))
            label3.place(x=240,y=230)

            label_4=Label(root,text="Location : ",width=20,font=("bold",13))
            label_4.place(x=80,y=280)
            label4 =Label(root,text=Location, width=20,font=("bold",13))
            label4.place(x=240,y=280)

            label_5=Label(root,text="Predicted Price : ",width=20,fg='red',font=("bold",13))
            label_5.place(x=90,y=330)
            input=pd.DataFrame([[Location,int(Area),int(Bathroom),int(BHK)]],columns=['location','total_sqft','bath','bhk'])
            op=pipe.predict(input)[0]
            ans = "{:.2f}".format(op)
            label5 =Label(root,text=ans, width=20,fg='red',font=("bold",13))
            label5.place(x=240,y=330)
        else:
            messagebox.showwarning("Warning","Area field accept only numeric value")
Button(root, text='Predicted Price' , width=20,bg="black",fg='white',command=predict).place(x=90,y=380)
Button(root, text='Quit' , width=20,bg="black",fg='white',command=root.destroy).place(x=240,y=380)
root.mainloop()




















