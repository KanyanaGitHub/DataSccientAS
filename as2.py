# ข้อที่ 1: ใช้ข้อมูล California Housing Dataset
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the California Housing dataset
cal_housing = fetch_california_housing()
df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
# Add the target variable to the DataFrame
df['target'] = cal_housing.target

# ข้อที่ 2: ทำความเข้าใจข้อมูล
print("ข้อมูลเบื้องต้น:")
print(df.info())
print("\nสถิติข้อมูล:")
print(df.describe())

# ข้อที่ 3: เตรียมข้อมูลด้วย dataframe
# ไม่มีขั้นตอนการเตรียมข้อมูลเพิ่มเติมที่กำหนด แต่เราสามารถทำการตรวจสอบ missing values ได้
print("\nตรวจสอบ missing values:")
print(df.isnull().sum())

# ข้อที่ 4: ตรวจสอบ Correlation และ boxplot
# สร้าง Correlation matrix
corr_matrix = df.corr()

# สร้างกราฟ heatmap เพื่อแสดง Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# สร้างกราฟ boxplot เพื่อแสดงการกระจายของฟีเจอร์ต่าง ๆ
plt.figure(figsize=(12, 8))
df.boxplot(rot=90)  # หมุน labels ที่แกน x ให้เอียง 90 องศา
plt.title("Boxplot of Features")
plt.show()

# ข้อที่ 5: ทดลอง 10-fold cross validation
# กำหนดตัวแปร X และ y โดย X คือฟีเจอร์ทั้งหมด และ y คือ target
X = df.drop('target', axis=1)
y = df['target']
# กำหนด KFold cross-validation ด้วยจำนวน folds = 10, shuffle=True เพื่อสลับลำดับข้อมูล, random_state=42 เพื่อทำให้ผลลัพธ์สามารถทำซ้ำได้
kf = KFold(n_splits=10, shuffle=True, random_state=42)
# สร้างโมเดล Linear Regression
model = LinearRegression()

# ทำการ cross-validation และคำนวณคะแนน R^2
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
print("\n10-Fold Cross Validation Scores (R^2):")
print(cv_scores)
print("\nMean R^2 Score:", cv_scores.mean())

# ข้อที่ 6: สร้างแบบจำลองทำนายค่า target
# ฝึกโมเดลด้วยข้อมูลทั้งหมด
model.fit(X, y)
# ทำการทำนายค่า target ด้วยโมเดลที่ฝึกแล้ว
y_pred = model.predict(X)
# ประเมินโมเดลด้วย Mean Squared Error (MSE) และ R^2 Score
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)