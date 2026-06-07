import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== 页面基础配置 =====================
st.set_page_config(
    page_title="Disease Risk Prediction Tool",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== 加载模型与数据 =====================
@st.cache_resource(show_spinner="Loading model, please wait...")
def load_model_resource():
    ann_model = pickle.load(open("ann_model.pkl", "rb"))
    feature_names = joblib.load("ann_feature_list.pkl")
    X_val = joblib.load("X_val.pkl")
    y_val = joblib.load("y_val.pkl")
    return ann_model, feature_names, X_val, y_val

model, feature_list, X_val, y_val = load_model_resource()

# ===================== 通用工具函数 =====================
# 风险分层
def strat_risk(p):
    if p < 0.3:
        return "Low-risk"
    elif p < 0.6:
        return "Intermediate-risk"
    else:
        return "High-risk"

# 批量表格风险列文字着色
def color_risk_text(val):
    if val == "Low-risk":
        return "color: green; font-weight: bold;"
    elif val == "Intermediate-risk":
        return "color: orange; font-weight: bold;"
    elif val == "High-risk":
        return "color: red; font-weight: bold;"
    return ""

# CSV下载缓存函数
@st.cache_data
def save_csv(d):
    return d.to_csv(index=False, encoding="utf-8-sig")

# ===================== 页面标题 =====================
st.title("Artificial Neural Network Disease Risk Prediction Calculator")
st.markdown("Input laboratory indicators to calculate individual disease risk")
st.divider()

# ===================== 左侧输入 + 右侧结果&图表（单例预测） =====================
col_left, col_right = st.columns([0.45, 0.55])

# 左侧：指标输入区
with col_left:
    st.subheader("Baseline Clinical Indicators")
    # 指标截断值说明
    cut_dict = {
        "WBC": "Cutoff=13.16, higher value indicates elevated risk",
        "FIB": "Cutoff=2.74, higher value indicates elevated risk",
        "NEU.": "Cutoff=47.20, higher value indicates elevated risk",
        "NLR": "Cutoff=1.01, higher value indicates elevated risk",
        "GBWT": "Cutoff=0.70, higher value indicates elevated risk"
    }

    input_list = []
    for var in feature_list:
        tip = cut_dict.get(var, "")
        st.markdown(f"<span style='font-size:17px; font-weight:bold;'>{var}</span> ({tip})", unsafe_allow_html=True)
        num = st.number_input("", value=0.0, step=0.01, key=var)
        input_list.append(num)

    calc_btn = st.button("Calculate Disease Risk", type="primary", use_container_width=True)

# 右侧：图表 + 预测结果
with col_right:
    st.subheader("Risk group stratifications of disease")
    st.markdown("Observed vs Predicted risk by decile groups")

    # 绘制十分位对比柱状图
    pred_val = model.predict_proba(X_val)[:, 1]
    df_dec = pd.DataFrame({"pred": pred_val, "y_true": y_val})
    df_dec["decile"] = pd.qcut(df_dec["pred"], q=10, labels=np.arange(1, 11), duplicates="drop")

    pred_mean = df_dec.groupby("decile")["pred"].mean().values
    obs_mean = df_dec.groupby("decile")["y_true"].mean().values
    x_axis = np.arange(1, 11)

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=300)
    bar_width = 0.35
    ax.bar(x_axis - bar_width/2, obs_mean, width=bar_width, label="Observed", color="#234b99")
    ax.bar(x_axis + bar_width/2, pred_mean, width=bar_width, label="Predicted", color="#89b8e8")
    ax.set_xlabel("Decile groups (10% quantile each)")
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Observed vs Predicted Risk Across Deciles")
    ax.set_xticks(x_axis)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    # 点击计算后展示个体风险
    if calc_btn:
        input_arr = np.array(input_list).reshape(1, -1)
        risk_prob = model.predict_proba(input_arr)[0, 1]
        risk_percent = risk_prob * 100
        risk_level = strat_risk(risk_prob)

        st.divider()
        st.markdown(f"### Predicted Disease Risk: **{risk_percent:.2f}%**")

        # 不同风险等级对应不同提示色
        if risk_level == "Low-risk":
            st.success(f"Risk Stratification: {risk_level}")
        elif risk_level == "Intermediate-risk":
            st.warning(f"Risk Stratification: {risk_level}")
        else:
            st.error(f"Risk Stratification: {risk_level}")

# ===================== 下方：批量预测（文件上传+彩色表格+下载） =====================
st.divider()
st.subheader("Batch Prediction via Uploaded Excel Dataset")
upload_file = st.file_uploader("Upload CSV / XLSX", type=["csv", "xlsx"])

if upload_file:
    # 兼容CSV / Excel
    if upload_file.name.endswith(".csv"):
        df_batch = pd.read_csv(upload_file)
    else:
        df_batch = pd.read_excel(upload_file)

    # 批量预测
    p_batch = model.predict_proba(df_batch)[:, 1]
    df_batch["Predicted_Risk"] = np.round(p_batch, 4)
    df_batch["Risk_Stratification"] = [strat_risk(i) for i in p_batch]

    # 表格列着色
    styled_df = df_batch.style.applymap(
        color_risk_text,
        subset=["Risk_Stratification"]
    )
    st.dataframe(styled_df, use_container_width=True)

    # 下载结果
    csv_data = save_csv(df_batch)
    st.download_button(
        label="Download Result CSV",
        data=csv_data,
        file_name="risk_prediction_results.csv",
        mime="text/csv"
    )
