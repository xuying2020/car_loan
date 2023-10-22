from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
import pandas as pd



with open('data_related.pkl', 'rb') as f:
    dataset = pickle.load(f)
print('原数据集维度：', dataset.shape)
seed = 2023
# 从数据集中分离出输入特征和目标变量
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


print("利用互信息选取前4个重要特征")
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
selector = SelectKBest(mutual_info_classif, k=4).fit(X_train, y_train)

# 获取每个特征的分数
feature_scores = selector.scores_
# 创建一个字典，将特征名和对应的分数进行配对
feature_contribution = dict(zip(X.columns, feature_scores))
# 打印所有变量的分数
for feature, score in feature_contribution.items():
    print(f"Feature: {feature}, Score: {score}")

X_new_train = selector.transform(X_train)
X_new_test = selector.transform(X_test)

X_new_train = pd.DataFrame(X_new_train, columns=X.columns[selector.get_support()])
X_new_test = pd.DataFrame(X_new_test, columns=X.columns[selector.get_support()])

# ------------------------利用SVM进行降采样----------------------------
# 训练 SVM 模型
from sklearn.svm import SVC
print("训练SVM")
svm = SVC(kernel='sigmoid', random_state=seed)
svm.fit(X_new_train, y_train)

# 提取支持向量的索引
support_idx = svm.support_
print("提取索引")
# 根据索引提取支持向量对应的样本
X_new_new_train = X_new_train.iloc[support_idx]
# 添加目标变量值
y_new_train = y_train.iloc[support_idx]

print('降维后数据集维度：', X_new_new_train.shape)

# ------------------------利用随机降采样----------------------------
from imblearn.under_sampling import RandomUnderSampler
# define undersample strategy 多数类和少数类的比例是 1:1
# undersample = RandomUnderSampler(sampling_strategy='majority')

# # define undersample strategy 可以调节多数类和少数类的比例
undersample = RandomUnderSampler(sampling_strategy=0.5)

# fit and apply the transform
X_under, y_under = undersample.fit_resample(X_new_train, y_train)

print('降维后数据集维度：', X_under.shape)


# ------------------------构建模型和模型评估----------------------------
clf = RandomForestClassifier(random_state=seed, n_estimators=200)
clf.fit(X_under, y_under)
y_pred = clf.predict(X_new_test)
report = classification_report(y_test, y_pred)
score = f1_score(y_test, y_pred, average='macro')
print('score:', round(score, 4))
print('Classification Report:\n', report)


# ------------------------误判分析----------------------------
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

y_true = y_test.values
C2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
print(C2)
print(C2.ravel())
# 显示热力图 GnBu
sns.heatmap(C2, annot=True, fmt='d', cmap="GnBu")
# 添加标题和标签
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
# 显示图像
plt.show()

import numpy as np
# 找出错误样本
clf = RandomForestClassifier(random_state=seed, n_estimators=200)
clf.fit(X_under, y_under)
y_pred = clf.predict(X_new_test)

selected_error_count = np.sum(np.abs(y_test-y_pred))  # 8720
selected_accuracy = 1-selected_error_count/float(y_test.shape[0])  # 0.7087897408495859

#region 绘制误判样本图
trees_num = 200
votes = np.round(clf.predict_proba(X_new_test) * trees_num).astype('int')
count_special1 = 0       #记录特殊样本总数
count_special2 = 0
special_sample1 = {}     #记录特殊的样本及其投票结果
special_sample2 = {}
test_index = list(pd.DataFrame(y_test).index)   #记录测试集中对应的行号
for i in range(len(test_index)):
    if y_pred[i] == list(y_test)[i]:
        pass
    else:
        if votes[i,0] < 50:
            count_special1 += 1
            special_sample1[i] = [votes[i, 0], votes[i, 1]]
        elif votes[i,0] > 150:
            count_special2 += 1
            special_sample2[i] = [votes[i, 0], votes[i, 1]]

#region 特殊样本分析
print('**********************************************************')
print('特殊样本分析：')
print('**********************************************************')
print('特殊样本1总数：',count_special1,'; 特殊样本2总数：',count_special2)  # 1864  1741
print('特殊样本1及其投票：',special_sample1)
print('特殊样本2及其投票',special_sample2)
rank1 = list(special_sample1.keys())
rank2 = list(special_sample2.keys())

special_sample1_order = [test_index[i] for i in rank1]  #抽取错误样本的行号
special_sample2_order = [test_index[i] for i in rank2]

train = pd.read_csv('car_loan_train.csv')
special_sample1_data = train.iloc[special_sample1_order,:]  #抽取错误样本的信息
special_sample2_data = train.iloc[special_sample2_order,:]

special_data = pd.concat([special_sample1_data,special_sample2_data],axis=0)
special_data.to_csv('special_data.csv')