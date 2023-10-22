from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import pandas as pd


with open('data_related.pkl', 'rb') as f:
    dataset = pickle.load(f)

print('原数据集维度：', dataset.shape)
seed = 2023
# 从数据集中分离出输入特征和目标变量
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

print("保留5个重要特征")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
selector = SelectKBest(f_classif, k=5).fit(X_train, y_train)  # 保留5个特征

# 获取被保留的特征的索引
selected_features = selector.get_support()
# 获取每个特征的分数
feature_scores = selector.scores_
# 创建一个字典，将特征名和对应的分数进行配对
feature_contribution = dict(zip(X.columns, feature_scores))
# 打印每个特征的贡献程度
for feature, score in feature_contribution.items():
    print(f"Feature: {feature}, Score: {score}")

X_new_train = selector.transform(X_train)
X_new_test = selector.transform(X_test)
X_new_train = pd.DataFrame(X_new_train, columns=X.columns[selector.get_support()])
X_new_test = pd.DataFrame(X_new_test, columns=X.columns[selector.get_support()])

# ------------------------利用SVM进行降采样----------------------------
# 训练 SVM 模型
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
# define undersample strategy 多数类和少数类比例1:1
# undersample = RandomUnderSampler(sampling_strategy='majority')

# # define undersample strategy 可以调节多数类和少数类的比例
undersample = RandomUnderSampler(sampling_strategy=0.5)

# fit and apply the transform
X_under, y_under = undersample.fit_resample(X_new_train, y_train)

print('降维后数据集维度：', X_under.shape)

# ------------------------构建模型和模型评估----------------------------
clf = RandomForestClassifier(random_state=seed, n_estimators=200)
clf.fit(X_new_new_train, y_new_train)
y_pred = clf.predict(X_new_test)
report = classification_report(y_test, y_pred)
score = f1_score(y_test, y_pred, average='macro')
print('score:', round(score, 4))
print('Classification Report:\n', report)

