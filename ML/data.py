
#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pandas.core.reshape.concat import concat
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

def get_csv(path):
    csv = pd.read_csv(path, sep=' ')
    csv.dropna(inplace=True)
    return mem_check(csv)

def get_decode(data, columns):
    for col in columns:
        data[f'{col}_num'] = data[col].map(data[col].value_counts())
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
        bin_list = [i*all_range/bin for i in range(all_range)]
        data[col + '_bin'] = pd.cut(data[col], bin_list, labels=False)
    return data

def get_cross_feature(data, cat_columns, num_columns):
    for f1 in cat_columns:
        g = data.groupby(f1, as_index=False)
        for f2 in tqdm(num_columns):
            feat = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max', '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
            })
            data = data.merge(feat, on=f1, how='left')
    return (data)

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

    #构建新的列
    for i in ['v_' +str(i) for i in range(14)]:
        for j in ['v_' +str(i) for i in range(14)]:
            all_data[str(i)+'+'+str(j)] = all_data[str(i)]+all_data[str(j)]
    
    name_list = ['model','brand', 'bodyType', 'fuelType','gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode']
    v_list = ['v_' +str(k) for k in range(14)]
    for name in name_list:
        print(name)
        for v in v_list:

            all_data[str(name)+'*'+str(v)] = all_data[name]*all_data[v]
            
    #重新设置时间
    all_data = get_date(all_data)

    need2decode = ['regDate', 'creatDate', 'model', 'brand', 'regionCode','bodyType','fuelType','name','regDate_year', 'regDate_month', 'regDate_day','regDate_dayofweek' , 'creatDate_month','creatDate_day', 'creatDate_dayofweek','kilometer'] 

    data = all_data.copy()
    data = get_decode(data, need2decode)

    #此处画汽车使用时间
    # 从注册时间到上线时间
    data['r2c_time'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') - 
                            pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days
    # 从注册时间到现在时间
    data['r2n_time'] = (pd.datetime.now() - pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days
    # 从上线时间到现在时间                        
    data['c2n_time'] = (pd.datetime.now() - pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') ).dt.days
    print(data)
    
    #对数据按照特定比例划分区间
    columns = ['power', 'r2c_time', 'r2n_time', 'c2n_time']
    data = get_region(data, columns, bin=50)

    #选取跟价格相关性高的特征获取交叉特征
    cross_cat = ['model', 'brand','regDate_year']
    cross_num = ['v_0','v_3', 'v_4', 'v_8', 'v_12','power']
    data = get_cross_feature(data, cross_cat, cross_num)

    #选择特征列
    columns = data.columns
    no_useful_feature = ['SaleID','offerType','seller']
    feature_columns = [col for col in columns if col not in no_useful_feature ]
    feature_columns = [col for col in feature_columns if col not in ['price']]
    X_train = data.iloc[:len(trainset),:][feature_columns]
    Y_train = trainset['price']
    X_test  = data.iloc[len(trainset):,:][feature_columns]

    train_data = [X_train, Y_train]
    test_data = [X_test]
    return train_data, test_data


if __name__ == '__main__':
    test_path = 'data/used_car_testB_20200421.csv'
    train_path = 'data/used_car_train_20200313.csv'
    dataManager = get_data(train_path, test_path)
