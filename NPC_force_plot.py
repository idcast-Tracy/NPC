# 打开网页，在cmd命令界面运行下面一段
# # streamlit run C:\Users\Tracy\Desktop\2024Winter\My_project\13.SHAP_Deploy\NPC_force_plot.py [ARGUMENTS]
## 终端运行
# # streamlit run NPC_force_plot.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# os.chdir(r'C:\Users\Tracy\Desktop\2024Winter\My_project\13.SHAP_Deploy') # 设定文件路径

st.set_page_config(page_title= 'Shapely加性解释',page_icon='⭐',layout='wide')
# st.title('SHAP 机器学习模型可解释')
a1 = st.selectbox('ICT',(0,1))
# a1 = st.radio('ICT',(0,1))
a2 = st.selectbox('EBV.DNA',(0,1))
# a2 = st.radio('EBV.DNA',(0,1))
a3 = st.number_input('Albumin')
a4 = st.number_input('riskScore')
features = pd.DataFrame({'ICT': [a1],'EBV.DNA': [a2],'Albumin': [a3],'riskScore': [a4]})



# def user_input_features():
#     st.sidebar.header('关键特征数值调整')
#     # a1 = st.sidebar.slider('',0.0,0.5,1.0) # 连续变量专用
#     # a3 = st.sidebar.radio('',(0,1)) # 分类变量专用
#     a1 = st.sidebar.radio('ICT',(0,1)) # 分类变量专用
#     a2 = st.sidebar.radio('EBV.DNA',(0,1)) # 分类变量专用
#     a3 = st.sidebar.slider('Albumin',0.0,0.5,1.0)
#     a4 = st.sidebar.slider('riskScore',0.0,0.5,1.0)
#     output = pd.DataFrame({'ICT': [a1],
#                            'EBV.DNA': [a2],
#                            'Albumin': [a3],
#                            'riskScore': [a4]})
#     return output
# features = user_input_features()

# ========================== XGBoost模型加载 ========================== #
import joblib
best_model = joblib.load('Model_Xgboost.pkl')

import shap
explainer = shap.TreeExplainer(best_model)


# ========================== 力图 ========================== #
if st.button('**Predict**'):
    st.set_option('deprecation.showPyplotGlobalUse', False) # 防止版本不兼容报错
    shap_values = explainer(features) # 保留三位小数
    shap_values.values = np.round(shap_values.values,3) # 保留三位小数
    shap_values.base_values = np.round(shap_values.base_values,3) # 保留三位小数
    shap_values.data = np.round(shap_values.data,3)
    # model_p = round(best_model.predict_proba(features)[0,1],3) # 分类用，预后不用
    # st.write('基于特征数值，得到该患者的shap_values.base_values:', shap_values.base_values)
    shap.plots.force(shap_values ,matplotlib=True ,show=False,text_rotation=15)
    st.pyplot(bbox_inches='tight')
    # shap.plots.waterfall(shap_values[0],show=False)
    # st.pyplot(bbox_inches='tight')
    st.write('**红色表示该特征的贡献是正数,蓝色表示该特征的贡献是负数**')
