import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN  # 随机采样函数 和SMOTE过采样函数
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pickle
import logging
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os

warnings.filterwarnings("ignore")

# 显示相关性高于0.6的变量
def getHighRelatedFeatureDf(corr_matrix, corr_threshold):
    highRelatedFeatureDf = pd.DataFrame(corr_matrix[corr_matrix > corr_threshold].stack().reset_index())
    highRelatedFeatureDf.rename({'level_0': 'feature1', 'level_1': 'feature2', 0: 'corr'}, axis=1, inplace=True)  # 更改列名
    highRelatedFeatureDf = highRelatedFeatureDf[
        highRelatedFeatureDf.feature1 != highRelatedFeatureDf.feature2]  # 去除自己和自己
    highRelatedFeatureDf['feature_pair_key'] = highRelatedFeatureDf.loc[:, ['feature1', 'feature2']].apply(
        lambda r: '#'.join(np.sort(r.values)), axis=1)
    # 将feature1和feature2名称连接在一起去重
    highRelatedFeatureDf.drop_duplicates(subset=['feature_pair_key'], inplace=True)
    highRelatedFeatureDf.drop(columns='feature_pair_key', inplace=True)
    return highRelatedFeatureDf


# 对原数据进行分析
def ori_data(X_train, X_test, y_train, y_test, seed=2023):
    print('原始标签训练集数据统计：', Counter(y_train))
    rf_model(X_train, X_test, y_train, y_test)


# 随机过采样方法
def random_oversampled(X_train, X_test, y_train, y_test, seed=2023):
    ros = RandomOverSampler(random_state=seed)  # random_state为0（此数字没有特殊含义，可以换成其他数字）使得每次代码运行的结果保持一致
    X_oversampled, y_oversampled = ros.fit_resample(X_train, y_train)  # 使用原始数据的特征变量和目标变量生成过采样数据集
    print('随机过采样处理后训练集数据统计', Counter(y_oversampled))

    rf_model(X_oversampled, X_test, y_oversampled, y_test)


# SMOTE法过采样
def smotesampled(X_train, X_test, y_train, y_test, seed=2023):
    smote = SMOTE(random_state=seed)  # random_state为0（此数字没有特殊含义，可以换成其他数字）使得每次代码运行的结果保持一致
    X_smotesampled, y_smotesampled = smote.fit_resample(X_train, y_train)  # 使用原始数据的特征变量和目标变量生成过采样数据集
    print('Smote法过采样后训练集数据统计', Counter(y_smotesampled))

    rf_model(X_smotesampled, X_test, y_smotesampled, y_test)


# KernelADASYN过采样
def adasynsampled(X_train, X_test, y_train, y_test, seed=2023):
    adasyn = ADASYN(random_state=seed)  # random_state为0（此数字没有特殊含义，可以换成其他数字）使得每次代码运行的结果保持一致
    X_adasynsampled, y_adasynsampled = adasyn.fit_resample(X_train, y_train)  # 使用原始数据的特征变量和目标变量生成过采样数据集
    print('KernelADASYN过采样', Counter(y_adasynsampled))

    rf_model(X_adasynsampled, X_test, y_adasynsampled, y_test)

from sklearn.metrics import f1_score
def rf_model(X_train, X_test, y_train, y_test, seed=2023):
    print('开始训练模型')
    clf = RandomForestClassifier(random_state=seed, n_estimators=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    score = f1_score(y_test, y_pred, average='macro')
    print('score:', round(score))
    print('Classification Report:\n', report)


def save_pkl(models, filename):
    '''Save models to a .pkl file'''
    with open(filename, 'wb') as file:
        pickle.dump(models, file)
    print(f"Models saved as {filename}")


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


if __name__ == '__main__':
    # ------------------读入数据-------------------------
    logging.info('data loading...')
    data = pd.read_csv('train_final.csv')
    print('原数据集维度：', data.shape)
    seed = 2023

    # 查找含有缺失值的列
    columns_with_missing_values = data.columns[data.isnull().any()]
    print(columns_with_missing_values)

    # 对'sub_Rate', 'main_Rate'列补0
    data['sub_Rate'].fillna(0, inplace=True)
    data['main_Rate'].fillna(0, inplace=True)

    # 删除'neighbor_default_prob', 'supplier_id_mean_target',
    #        'employee_code_id_mean_target'中的缺失值样本
    data.dropna(inplace=True)

    # ------------------相关性分析-------------------------
    print('相关性分析前维度', data.shape)
    # 删除分类特征
    data_related = data.drop(['branch_id', 'supplier_id', 'manufacturer_id', 'area_id',
                              'employee_code_id', 'Credit_level', 'loan_default'], axis=1)
    related = getHighRelatedFeatureDf(data_related.corr(), 0.9)
    # print('相关性大于0.9的特征', related)

    # 删除相关性高于0.9的特征
    data.drop(['main_account_disbursed_loan', 'total_account_loan_no',  'sub_account_inactive_loan_no',
               'main_account_tenure',  'loan_to_asset_ratio',  'main_account_inactive_loan_no',
               'main_account_monthly_payment',  'main_account_overdue_no',  'main_account_outstanding_loan',
               'main_account_sanction_loan'], axis=1, inplace=True)

    print('相关性分析后维度', data.shape)

    save_pkl(data, 'data_related.pkl')

    print('数据保存成功')


    # # 分别获取特征值和标签值
    # X = data.drop('loan_default', axis=1)
    # y = data['loan_default']
    #
    #
    # # 将数据集划分为训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    #
    # rf_model(X_train, X_test, y_train, y_test)
