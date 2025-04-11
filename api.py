from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import io

# 创建 FastAPI 应用实例
app = FastAPI(title="🧠 PyCaret ML API")

# 加载保存好的 PyCaret 模型
model = joblib.load("best_pipeline.pkl")  # 注意模型文件名必须一致

# API 首页测试用


@app.get("/")
def root():
    return {"message": "🎯 FastAPI 服务已启动成功"}

# 上传 CSV 文件进行预测


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 读取上传的 CSV 文件内容
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # 调用模型进行预测
    preds = model.predict(df)
    df["prediction"] = preds

    # 返回前 10 条预测结果（JSON 格式）
    return df.head(10).to_dict(orient="records")
