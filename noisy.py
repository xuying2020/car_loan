from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from cleanlab.classification import CleanLearning
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab.internal.util import value_counts
from cleanlab.internal.latent_algebra import compute_inv_noise_matrix
import pandas as pd


def results(cl):
    print("没有使用置信学习，标签中含有噪声", end=" ")
    clf = RandomForestClassifier(n_estimators=500,criterion='entropy')
    _ = clf.fit(X_train, s)
    pred = clf.predict(X_test)
    print("dataset test accuracy:", round(accuracy_score(pred, y_test), 2))

    print("\n现在我们展示了使用 cleanlab 表征噪声的改进并学习（具有高置信度）正确标记的数据。")
    print()

    print("使用置信学习，给出噪声矩阵，标签中含有噪声,", end=" ")
    _ = cl.fit(X_train, s, noise_matrix=noise_matrix)
    pred = cl.predict(X_test)
    print("dataset test accuracy:", round(accuracy_score(pred, y_test), 2))
    print()

    print("使用置信学习，给出噪声/反噪声矩阵，标签中含有噪声,", end=" ")
    inv = compute_inv_noise_matrix(py, noise_matrix)
    _ = cl.fit(X_train, s, noise_matrix=noise_matrix, inverse_noise_matrix=inv)
    pred = cl.predict(X_test)
    print("dataset test accuracy:", round(accuracy_score(pred, y_test), 2))
    print()

    print("使用置信学习，噪声矩阵未给出，标签中含有噪声", end=" ")
    _ = cl.fit(X_train, s)
    pred = cl.predict(X_test)
    print("dataset test accuracy:", round(accuracy_score(pred, y_test), 2))


if __name__ == '__main__':
    # Seed for reproducibility
    seed = 2
    clf = RandomForestClassifier(n_estimators=500,criterion='entropy')
    cl = CleanLearning(clf=clf, seed=seed, verbose=False)
    np.random.seed(seed=seed)

    # Get iris dataset
    # 读入数据
    data = pd.read_csv('car_loan_train.csv')
    # 找到包含inf的行索引
    inf_index = data.index[data.isin([np.inf, -np.inf]).any(1)]
    # 删除这些行
    data.drop(inf_index, inplace=True)
    row = ['main_account_loan_no',	'main_account_active_loan_no',	'main_account_overdue_no',	'main_account_outstanding_loan',	'main_account_sanction_loan',	'main_account_disbursed_loan',	'sub_account_loan_no',	'sub_account_active_loan_no',	'sub_account_overdue_no',	'sub_account_outstanding_loan',	'sub_account_sanction_loan',	'sub_account_disbursed_loan',	'disbursed_amount',	'asset_cost',	'branch_id',	'supplier_id',	'manufacturer_id',	'area_id',	'employee_code_id',	'Driving_flag',	'passport_flag',	'credit_score',	'main_account_monthly_payment',	'sub_account_monthly_payment',	'last_six_month_new_loan_no',	'last_six_month_defaulted_no',	'average_age',	'credit_history',	'enquirie_no',	'loan_to_asset_ratio',	'total_account_loan_no',	'sub_account_inactive_loan_no',	'total_inactive_loan_no',	'main_account_inactive_loan_no',	'total_overdue_no',	'total_outstanding_loan',	'total_sanction_loan',	'total_disbursed_loan',	'total_monthly_payment',	'outstanding_disburse_ratio',	'main_account_tenure',	'sub_account_tenure',	'disburse_to_sactioned_ratio',	'active_to_inactive_act_ratio',	'Credit_level',	'employment_type',	'age',]
    X = data[row].values
    y = data['loan_default'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # iris = datasets.load_iris()
    # X = iris.data  # we only take the first two features.
    # y = iris.target
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Generate lots of noise.
    noise_matrix = np.array(
        [
            [0.7, 0.2],
            [0.3, 0.8],
        ]
    )
    # - 类别0内正确分类的概率为0.7
    # - 类别1内正确分类的概率为0.8
    # - 类别0被误分类到类别1的概率为0.3
    # - 类别1被误分类到类别0的概率为0.2
    # 矩阵行和为1, 表示所有的分类结果的概率和为1。
    # 这个2x2噪声矩阵可以用于二分类问题, 生成带噪声的标签, 例如:
    # 如果原始标签是:
    # y_train = [0, 0, 1, 1]
    # 用这个noise_matrix可以生成的噪声标签是:
    # s = [0, 1, 1, 0]
    py = value_counts(y_train)

    # Create noisy labels
    s = generate_noisy_labels(y_train, noise_matrix)
    results(cl)



# 没有使用置信学习，标签中含有噪声 dataset test accuracy: 0.78
#
# 现在我们展示了使用 cleanlab 表征噪声的改进并学习（具有高置信度）正确标记的数据。
#
# 使用置信学习，给出噪声矩阵，标签中含有噪声, dataset test accuracy: 0.79
#
# 使用置信学习，给出噪声/反噪声矩阵，标签中含有噪声, dataset test accuracy: 0.8
#
# 使用置信学习，噪声矩阵未给出，标签中含有噪声 dataset test accuracy: 0.8