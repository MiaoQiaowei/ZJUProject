
#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pandas.core.reshape.concat import concat
import warnings
warnings.filterwarnings('ignore')
def get_csv(path):
    csv = pd.read_csv(path, sep=' ')
    csv.dropna(inplace=True)
    return mem_check(csv)

def get_decode(data, columns):
    for col in columns:
        data[f'{col}_num'] = data[col].map(data[col].value_counts)
    return data 

def type_trans(x):
    int_x = int(x[4:6])
    if int_x == 0:
        int_x = 1
    return x[:4] + '-' + str(int_x) + '-' + x[6:]

def get_date(data, columns=['regDate', 'creatDate']):
    for col in columns:
        data[col] = pd.to_datetime(data[col].astype('str').apply(type_trans))
        data[col + '_year'] = data[col].dt.year
        data[col + '_month'] = data[col].dt.month
        data[col + '_day'] = data[col].dt.day
        data[col + '_dayofweek'] = data[col].dt.dayofweek
    return (data)

def  get_region(data, columns, bin=50):
    for col in columns:
        all_range = int(data[col].max()-data[col].min())
        bin = [i*all_range/bin for i in range(all_range)]
        data[col + '_bin'] = pd.cut(data[col], bin, labels=False)
    return data

def mem_check(df):
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    return df

def get_data(train_path, test_path):
        trainset = get_csv(train_path)
        testset = get_csv(test_path)
        all_data = pd.concat([trainset, testset])
        all_data['notRepairedDamage'] = all_data['notRepairedDamage'].replace('-',0).astype('float16')
        common_value = all_data.mode().iloc[0,:]
        all_data = all_data.fillna(common_value)
        

        #将数据限定到特定区间
        print(all_data)
        # 此处画功率power的图
        print(all_data['power'].max())
        print(all_data['power'].min())
        
        all_data[all_data['power']>600]['power'] = 600
        all_data[all_data['power']<1]['power'] = 1

        all_data[all_data['v_13']>6]['v_13'] = 6
        all_data[all_data['v_14']>4]['v_14'] = 4
        print("!!"*100)
        print(all_data['regDate'])
        all_data = get_date(all_data)

        #构建新的列
        for i in ['v_' +str(i) for i in range(14)]:
            for j in ['v_' +str(i) for i in range(14)]:
                all_data[str(i)+'+'+str(j)] = all_data[str(i)]+all_data[str(j)]
        for i in ['model','brand', 'bodyType', 'fuelType','gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode']:
            for j in ['v_' +str(i) for i in range(14)]:
                all_data[str(i)+'*'+str(j)] = all_data[i]*all_data[j]

        need2decode = ['regDate', 'creatDate', 'model', 'brand', 'regionCode','bodyType','fuelType','name','regDate_year', 'regDate_month', 'regDate_day','regDate_dayofweek' , 'creatDate_month','creatDate_day', 'creatDate_dayofweek','kilometer'] 

        data = all_data.copy()
        data = get_decode(data, need2decode)

        #此处画汽车使用时间
        
        print(data)
    




if __name__ == '__main__':
    test_path = 'data/used_car_testB_20200421.csv'
    train_path = 'data/used_car_train_20200313.csv'
    dataManager = get_data(train_path, test_path)
