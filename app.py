import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="ANN Disease Risk Prediction Tool", layout="wide")

@st.cache_resource(show_spinner="Loading model...")
def load_model_resource():
    ann_model = pickle.load(open("ann_model.pkl", "rb"))
    feature_names = joblib.load("ann_feature_list.pkl")
    X_val = joblib.load("X_val.pkl")
    y_val = joblib.load("y_val.pkl")
    return ann_model, feature_names, X_val, y_val

model, feature_list, X_val, y_val = load_model_resource()

st.title("Artificial Neural Network Disease Risk Prediction Calculator")
st.markdown("Input laboratory indicators to calculate individual disease risk; cutoff values derived from SHAP analysis.")
st.divider()

# 改成左右对半均分 0.5:0.5
col_left, col_right = st.columns([0.5, 0.5])

with col_left:
    st.subheader("Baseline Clinical Indicators")
    cut_dict = {
        "WBC": "Cutoff=13.16, higher value indicates elevated risk",
        "FIB": "Cutoff=2.74, higher value indicates elevated risk",
        "NEU.": "Cutoff=47.20, higher value indicates elevated risk",
        "NLR": "Cutoff=1.01, higher value indicates elevated risk",
        "GBWT": "Cutoff=0.70, higher value indicates elevated risk"
    }
    input_list = []
    for var in feature_list:
        tip = cut_dict[var]
        st.markdown(f"<span style='font-size:17px; font-weight:bold;'>{var} {tip}</span>", unsafe_allow_html=True)
        num = st.number_input("", value=0.0, step=0.01, key=var)
        input_list.append(num)
    calc_btn = st.button("Calculate Disease Risk", type="primary")

with col_right:
    st.subheader("Risk group stratifications of disease")
    if calc_btn:
        df_in = pd.DataFrame([input_list], columns=feature_list)
        risk_all = model.predict_proba(df_in)[0][1]
        st.markdown(f"**The predicted risk to develop disease: {risk_all:.2%}**")
        st.divider()

    st.subheader("Observed vs Predicted risk by decile groups")
    pred_val = model.predict_proba(X_val)[:,1]
    df_dec = pd.DataFrame({"pred":pred_val, "y_true":y_val})
    df_dec["decile"] = pd.qcut(df_dec["pred"], q=10, labels=np.arange(1,11))
    pred_mean = df_dec.groupby("decile")["pred"].mean().values
    obs_mean = df_dec.groupby("decile")["y_true"].mean().values
    x_axis = np.arange(1,11)

    plt.rcParams["font.family"] = "Arial"
    fig,ax = plt.subplots(figsize=(7,3.8),dpi=300)
    w = 0.35
    ax.bar(x_axis-w/2, obs_mean, width=w, label="Observed", color="#234b99")
    ax.bar(x_axis+w/2, pred_mean, width=w, label="Predicted", color="#89b8e8")
    ax.set_xlabel("Decile groups (10% quantile each)")
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Observed vs Predicted Risk Across Deciles")
    ax.set_xticks(x_axis)
    ax.legend()
    st.pyplot(fig)

st.divider()
st.subheader("Batch Prediction via Uploaded Excel Dataset")
up = st.file_uploader("Upload CSV / XLSX", type=["csv","xlsx"])
if up:
    df_batch = pd.read_excel(up)
    p_batch = model.predict_proba(df_batch)[:,1]
    df_batch["Predicted_Risk"] = np.round(p_batch,4)
    def strat_risk(p):
        if p<0.3:return "Low-risk"
        elif p<0.6:return "Intermediate-risk"
        else:return "High-risk"
    df_batch["Risk_Stratification"] = [strat_risk(i) for i in p_batch]
    st.dataframe(df_batch, use_container_width=True)
    @st.cache_data
    def save_csv(d):
        return d.to_csv(index=False,encoding="utf-8-sig")
    st.download_button("Download Result CSV", data=save_csv(df_batch), file_name="Cohort_Risk_Output.csv")