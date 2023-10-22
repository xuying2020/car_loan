# 画图看降到多少维合适
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt

with open('data_related.pkl', 'rb') as f:
    dataset = pickle.load(f)
print('原数据集维度：', dataset.shape)
seed = 2023
# 从数据集中分离出输入特征和目标变量
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

candidate_components = range(1, 38, 1)  # 有50个特征，我们最低取5维，并以5为步长递增
explained_ratios = []
for c in candidate_components:
    pca = PCA(n_components=c)
    pca.fit(X_train)
    explained_ratios.append(np.sum(pca.explained_variance_ratio_))
print(explained_ratios)

plt.figure(figsize=(10, 6), dpi=144)
plt.grid()
plt.plot(candidate_components, explained_ratios)
plt.xlabel('Number of PCA Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained variance ratio for PCA')
plt.yticks(np.arange(0.5, 1, 0.1))
plt.xticks(np.arange(0, 50, 5))
plt.show()

pca = PCA(n_components=2)
X_new_train = pca.fit_transform(X_train)
X_new_test = pca.transform(X_test)

X_new_train = pd.DataFrame(X_new_train)
X_new_test = pd.DataFrame(X_new_test)

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
# define undersample strategy  多数类和少数类比例1:1
# undersample = RandomUnderSampler(sampling_strategy='majority')

# # define undersample strategy  可以调节多数类和少数类的比例
undersample = RandomUnderSampler(sampling_strategy=0.35)

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
