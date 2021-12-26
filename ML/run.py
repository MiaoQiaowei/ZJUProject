import pandas as pd
from DNN import *
from data import get_data
from encoder import MeanEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from tqdm import tqdm

if __name__ == '__main__':
    test_path = 'data/used_car_testB_20200421.csv'
    train_path = 'data/used_car_train_20200313.csv'
    trainset, testset = get_data(train_path, test_path)
    data, label = trainset
    

    encode_list = ['regDate', 'creatDate','model','brand','name','regionCode']
    Encoder = MeanEncoder(encode_list, target_type='regression')
    data = Encoder.fit_transform(data, label)
    test = Encoder.transform(testset[0])
    data['price'] = label
    
    print(data.columns)
    # 对目标编码
    encode_list = []
    price_dict = {
        'max': data['price'].max(),
        'min': data['price'].min(),
        'median': data['price'].median(),
        'mean': data['price'].mean(),
        'sum': data['price'].sum(),
        'std': data['price'].std(),
        'skew': data['price'].skew(),
        'kurt': data['price'].kurt(),
        'mad': data['price'].mad()
    }
    
    # k折线计算
    price_encode = ['max', 'min', 'mean']
    KFold_func = KFold(n_splits=10, shuffle=True, random_state=42)
    feature_list = ['regionCode','brand','regDate_year','creatDate_year','kilometer','model']
    for f in  tqdm(feature_list):
        encode_dict = {}
        for pe in price_encode:
            encode_dict['{}_target_{}'.format(f, pe)] = pe
            data['{}_target_{}'.format(f, pe)] = 0
            test['{}_target_{}'.format(f, pe)] = 0
            encode_list.append('{}_target_{}'.format(f, pe))
        for i, (train_idx, val_idx) in enumerate(KFold_func.split(data, label)):
            train_x, val_x = data.iloc[train_idx].reset_index(drop=True), data.iloc[val_idx].reset_index(drop=True)
            encode_data = train_x.groupby(f, as_index=False)['price'].agg(encode_dict)
            val_x = val_x[[f]].merge(encode_data, on=f, how='left')
            test_x = test[[f]].merge(encode_data, on=f, how='left')
            for pe in price_encode:
                #对特征进行填充
                val_x['{}_target_{}'.format(f, pe)] = val_x['{}_target_{}'.format(f, pe)].fillna(price_dict[pe])
                test_x['{}_target_{}'.format(f, pe)] = test_x['{}_target_{}'.format(f, pe)].fillna(price_dict[pe])
                data.loc[val_idx, '{}_target_{}'.format(f, pe)] = val_x['{}_target_{}'.format(f, pe)].values 
                test['{}_target_{}'.format(f, pe)] += test_x['{}_target_{}'.format(f, pe)].values / KFold_func.n_splits

    print(len(data.columns))
    #删除没有用的列，此处可重新画图
    need2drop = ['regDate', 'creatDate','brand_power_min', 'regDate_year_power_min']
    data = data.drop(need2drop+['price'],axis=1)
    test = test.drop(need2drop,axis=1)
    data = data.astype('float32')
    test = test.astype('float32')
    
    print(data)
    print(test)

    # 对特征进行归一化
    # min_max_scaler = MinMaxScaler()
    # tmp = pd.concat([data,test])
    # tmp = tmp.dropna()
    # min_max_scaler.fit(tmp.values)
    # all_data = min_max_scaler.transform(tmp.values)
    # print(all_data)
    # pca = decomposition.PCA(n_components=146)
    # all_pca = pca.fit_transform(all_data)
    # X_pca = all_pca[:len(data)]
    # test = all_pca[len(data):]
    # y = label.values
    y = np.load('label.npy')
    train = np.load('train.npy')
    test = np.load('test.npy')

    def scheduler(epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 20 == 0 and epoch != 0:
            lr = K.backend.get_value(model.optimizer.lr)
            K.backend.set_value(model.optimizer.lr, lr * 0.6)
            print("lr changed to {}".format(lr * 0.6))
        return K.backend.get_value(model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)

    n_splits = 6
    KF = KFold(n_splits=n_splits, shuffle=True)

    b_size = 2000
    max_epochs = 100
    oof_pred = np.zeros((len(train), ))

    sub = pd.read_csv('./data/used_car_testB_20200421.csv',sep = ' ')[['SaleID']].copy()
    sub['price'] = 0

    avg_mae = 0
    for fold, (train_idx, val_idx) in enumerate(KF.split(train, y)):
        print('fold:', fold)
        X_train, y_train = train[train_idx], y[train_idx]
        X_val, y_val = train[val_idx], y[val_idx]
        
        model = DNN(X_train.shape[1])
        simple_adam = tf.optimizers.Adam(lr = 0.015)
        
        model.compile(loss='mae', optimizer=simple_adam,metrics=['mae'])
        es = EarlyStopping(monitor='val_score', patience=10, verbose=2, mode='min', restore_best_weights=True,)
        es.set_model(model)
        metric = Metric(model, [es], ((X_train, y_train), (X_val, y_val)))
        model.fit(X_train, y_train, batch_size=b_size, epochs=max_epochs, 
                validation_data = (X_val, y_val),
                callbacks=[reduce_lr], shuffle=True, verbose=2)
        y_pred3 = model.predict(X_val)
        y_pred = np.zeros((len(y_pred3), ))
        sub['price'] += model.predict(test).reshape(-1,)/n_splits
        for i in range(len(y_pred3)):
            y_pred[i] = y_pred3[i]
            
        oof_pred[val_idx] = y_pred
        val_mae = mean_absolute_error(y[val_idx], y_pred)
        avg_mae += val_mae/n_splits
        print('val_mae is:{}'.format(val_mae))
    mean_absolute_error(y, oof_pred)
    sub.to_csv('nn_sub_{}_{}.csv'.format('mae', sub['price'].mean()), index=False)