import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

# 读入数据
data = pd.read_csv('car_loan_train.csv')
# 查看数据前3行
data.head(3)
# 查看数据总体情况
data.describe()  # 给出样本数据的一些基本的统计量，包括均值，标准差，最大值，最小值，分位数等
data.info()  # 给出样本数据的相关信息概览 ：行数，列数，列索引，列非空值个数，列类型，内存占用
data.shape  # 查看行数和列数   --150000*53
data[['disbursed_amount']].describe()  # 查看指定列的统计信息
print('loan_default：\n', (data['loan_default'].value_counts() / len(data)).round(2))  # 统计目标变量比例  0.82:0.18
# ------------------查看数据的缺失值比例  --无缺失值
data.isna().mean()
print('缺失值数：\n', data.isnull().any(axis=1).sum())
# # ------------------查看异常值
# # 区分数值变量和分类变量
Ca_feature = list(data[['Driving_flag', 'passport_flag', ]])
Nu_feature = list(data[['disbursed_amount', 'asset_cost', 'credit_score', 'average_age', 'credit_history',
                        'loan_to_asset_ratio', 'active_to_inactive_act_ratio', 'age', ]])  # 数值变量
# 绘制箱线图
plt.figure(figsize=(30, 15))  # 箱线图查看数值型变量异常值
i = 1
for col in Nu_feature:
    ax = plt.subplot(2, 4, i)
    ax = sns.boxplot(x="loan_default", y=col, data=data[[col, 'loan_default']],
                     boxprops={'color': '#508CB4', 'facecolor': '#508CB4'})
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    i += 1
# plt.show()
plt.savefig("boxplot.png", bbox_inches='tight')
# # 发现存在变量有异常 由于是特殊的风险预测，所以保留异常值
#
# 分类数据
data['year_of_birth'].value_counts()
data['employment_type'].value_counts()
data['Credit_level'].value_counts()
# # 堆积柱状图
plt.figure(figsize=(12, 7))
rgbcolor = [(80 / 255, 140 / 255, 180 / 255), (215 / 255, 228 / 255, 239 / 255)]
i = 1
for col in Ca_feature:
    ax = plt.subplot(1, 2, i)
    df_group = pd.crosstab(data[col], data['loan_default'])
    labels = df_group.index.values
    fraud0 = []
    fraud1 = []
    for j in range(len(labels)):
        fraud0.append(df_group.iat[j, 0])
        fraud1.append(df_group.iat[j, 1])
    width = 0.5
    ax.bar(labels, fraud0, width, label='default0', color=rgbcolor[0])
    ax.bar(labels, fraud1, width, bottom=fraud0, label='default1', color=rgbcolor[1])
    ax.xaxis.set_major_locator(mticker.FixedLocator(labels))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
    ax.set_xticklabels(labels, rotation=20)
    ax.legend()
    ax.set_title(col)
    i = i + 1
plt.tight_layout()
plt.show()
# plt.savefig("Ca_feature.png", bbox_inches='tight')

# 分类超过10个的单独作图 'manufacturer_id','area_id', 'branch_id', 'year_of_birth','
plt.figure(figsize=(30, 10))
ax = plt.subplot()
rgbcolor = [(80 / 255, 140 / 255, 180 / 255), (215 / 255, 228 / 255, 239 / 255)]
df_group = pd.crosstab(data['age'], data['loan_default'])
labels = df_group.index.values
fraud0 = []
fraud1 = []
for j in range(len(labels)):
    fraud0.append(df_group.iat[j, 0])
    fraud1.append(df_group.iat[j, 1])
width = 0.5
ax.bar(labels, fraud0, width, label='default0', color=rgbcolor[0])
ax.bar(labels, fraud1, width, bottom=fraud0, label='default1', color=rgbcolor[1])
ax.xaxis.set_major_locator(mticker.FixedLocator(labels))
ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
ax.set_xticklabels(labels, rotation=20)
ax.legend()
ax.set_title('age')
plt.show()
plt.tight_layout()
plt.savefig("age.png", bbox_inches='tight')

# 删除customer_id,year_of_brith,disbursed_date,mobileno_flag,idcard_flag
# 删除 outstanding_disburse_ratio 特征中 值=inf 的样本
# 得到oriData

# --------------------相关性分析
numercial_feature4 = ['loan_to_asset_ratio',
                      'total_disbursed_loan',
                      'main_account_disbursed_loan',
                      'total_sanction_loan',
                      'main_account_sanction_loan',
                      'total_outstanding_loan',
                      'total_account_loan_no',
                      'main_account_loan_no',
                      'main_account_monthly_payment',
                      'main_account_inactive_loan_no',
                      'main_account_overdue_no',
                      'main_account_tenure',
                      'main_account_outstanding_loan',
                      'sub_account_inactive_loan_no',
                      'total_monthly_payment',
                      'total_overdue_no']
cor = data[numercial_feature4].corr()
sns.set_theme(style="white")
plt.figure(figsize=(20, 20))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(cor, cmap=cmap, annot=True, linewidth=0.2,
            cbar_kws={"shrink": .5}, linecolor="white", fmt=".1g")
plt.xticks(rotation=75)
plt.savefig("heatmap4.png", bbox_inches='tight')
# plt.show()




