import os,gc
from collections import defaultdict
from sklearn.metrics import mean_absolute_error,precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, accuracy_score


def split_data_withidx(n_fold, df):
    train_k = [[] for _ in range(n_fold)]  # k=5 (validation - 14,15,16,17,18)

    year = [2014, 2015, 2016, 2017, 2018]

    for k,i in enumerate(year):
        idx = list(total[total['year'] == i].index)
        train_k[k] = idx

    return train_k



def get_fold_data(idx, data, train_k):
    
    train = data[data['id'].isin(train_k[idx]) == False]  # train
    val = data[data['id'].isin(train_k[idx])] # validation

    return train, val



def lgb_kfold_prediction(train, test, FEATS,  model_params=None, folds=None ):  # categorical_features='auto',

    # fold 생성
    train_k = split_data_withidx(folds, train)
    
    x_test = test[FEATS] 
    
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    # acc=0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = FEATS


    for fold in range(folds):

        # train index, validation index로 train 데이터를 나눔
        train_set, valid_set = get_fold_data(fold, train, train_k) 
   

        x_tr, x_val = train_set[FEATS], valid_set[FEATS]   
        y_tr, y_val = train_set['공급량'], valid_set['공급량']  # 정답 부분
    
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # LightGBM 데이터셋 선언
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val)
        
        # LightGBM 모델 훈련
        model = lgb.train(
            model_params,
            dtrain,
            valid_sets=[dtrain, dvalid], # Validation 성능을 측정할 수 있도록 설정
            # categorical_feature='auto',
            verbose_eval=20,    
            num_boost_round=500,
            early_stopping_rounds=100
        )

        # model = lgb.train(params, d_train, 500, d_val, verbose_eval=20, early_stopping_rounds=10)


        # Validation 데이터 예측
        val_preds = model.predict(x_val)
        
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | MAE: {mean_absolute_error(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += mean_absolute_error(y_val, val_preds) / folds
#         acc += accuracy_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += model.predict(x_test) / folds
        
        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = model.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean MAE = {score} ") # 폴드별 Validation 스코어 출력

    
    
    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return test_preds, fi 
