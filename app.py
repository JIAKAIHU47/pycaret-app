import streamlit as st
import pandas as pd
import joblib

# 加载模型
model = joblib.load("best_pipeline.pkl")

# 页面标题
st.set_page_config(page_title="🧠 PyCaret 预测系统", layout="centered")
st.title("📊 PyCaret 模型预测界面")

# 上传 CSV
file = st.file_uploader("请上传测试数据 CSV 文件", type=["csv"])

if file:
    try:
        df = pd.read_csv(file)
        st.subheader("📝 输入数据预览：")
        st.dataframe(df.head())

        # 模型预测
        preds = model.predict(df)
        df["prediction"] = preds

        st.subheader("✅ 预测结果：")
        st.dataframe(df.head())

        # 提供下载
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载预测结果 CSV",
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"❌ 出错了：{e}")
else:
    st.info("👈 请上传一个 CSV 文件进行预测")
