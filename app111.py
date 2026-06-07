
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import pickle
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample
import seaborn as sns
from scipy.stats import norm
from sklearn.calibration import calibration_curve
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score
 
exit()


pip install streamlit scikit-learn pandas numpy joblib --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple

  streamlit run app.py
####################### 一、数据导入及预处理 ########

os.getcwd() # 当前工作路径
plt.rcParams['font.family'] = 'Times New Roman' # 设置字体为Times New Roman，与r统一
# 导入数据
train_data = pd.read_csv("traindata.csv", encoding="GBK")   
val_data = pd.read_csv("valdata.csv", encoding="GBK")

train_data_ml = pd.read_csv("traindata.csv", encoding="GBK")    # 想用过采样的训练集的话需要改名字⭐⭐⭐
val_data_ml = pd.read_csv("valdata.csv", encoding="GBK")
X_train = train_data_ml[['WBC', 'FIB', 'NEU.', 'NLR', 'GBWT']] # 批量定义自变量
y_train = train_data_ml['PATH'] 
X_val = val_data_ml[['WBC', 'FIB', 'NEU.', 'NLR', 'GBWT']] # 批量定义自变量
y_val = val_data_ml['PATH'] 

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'





ann_default = MLPClassifier(random_state=123, max_iter=500)
ann_default.fit(X_train, y_train)
# 计算默认参数模型的验证集AUC
y_val_pred_prob_annd = ann_default.predict_proba(X_val)[:, 1]
auc_annd = roc_auc_score(y_val, y_val_pred_prob_annd)
print("默认参数ANN模型的验证集 AUC:", auc_annd)

# 定义超参数搜索范围
param_grid = {
    'hidden_layer_sizes': [(25,), (50,), (100,), (10, 10), (50, 50), (100, 100), (50, 50, 50)],  # 隐藏层神经元数
    'activation': ['relu', 'tanh', 'logistic']  # 激活函数
}

# 使用 GridSearchCV 进行网格搜索和 k 折交叉验证
grid_search_ann = GridSearchCV(
    estimator=ann_default,  # 使用之前定义的默认参数模型
    param_grid=param_grid,  # 使用之前定义的超参数网格
    scoring='roc_auc',  # 使用 AUC 作为评价指标
    cv=5,  # 5 折交叉验证
    n_jobs=-1,  # 并行计算
    verbose=1
)

grid_search_ann.fit(X_train, y_train)  # 在训练集上进行网格搜索

# 输出最优超参数组合
best_auc_ann = grid_search_ann.best_score_  # 获取最佳模型的交叉验证 AUC
ann_model_best = grid_search_ann.best_estimator_  # 获取最佳模型
best_params_ann = grid_search_ann.best_params_  # 获取最佳超参数组合
print("最佳ANN参数组合:", best_params_ann)
print("调优ANN模型详细参数:", pd.DataFrame.from_dict(ann_model_best.get_params(), orient='index'))
print("默认参数ANN模型的验证集 AUC:", auc_annd)
print("参数调优ANN模型最佳模型的交叉验证 AUC:", best_auc_ann)
# 保存训练好的模型
with open("2.训练集构建模型/ann_model.pkl", 'wb') as f:
    pickle.dump(ann_model_best, f)
    import joblib
# X_train是你的训练自变量，保存特征列表
joblib.dump(list(X_train.columns), "ann_feature_list.pkl")
import joblib
joblib.dump(X_val,"X_val.pkl")
joblib.dump(y_val,"y_val.pkl")

 

