import numpy as np
import pandas as pd
import csv #for exel
import openpyxl #for exel
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from html import escape #for exel

# ดาวน์โหลดข้อมูล
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# เเยกข้อมูล
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# loadข้อมูลเข้าในpandas DataFrame
data = pd.read_csv(url, header=None, names=columns)

# แยก features และ labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# สร้าง StandardScaler สำหรับ normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# สร้างโมเดล Naïve Bayes และ kNN
nb_model = GaussianNB()
knn_model = KNeighborsClassifier(n_neighbors=5)

# จำนวนรอบการทดลอง
n_experiments = 30

# เก็บผลลัพธ์
nb_results = []
knn_results = []

for i in range(1, n_experiments + 1):
    # เปลี่ยนค่า seed number ในแต่ละครั้ง (จาก 1 ถึง 30)
    seed = i
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    
    # ประเมินโมเดล Naïve Bayes ด้วย cross-validation
    nb_scores = cross_val_score(nb_model, X_scaled, y, cv=kf, scoring='accuracy')
    nb_results.append(nb_scores.mean())
    
    # ประเมินโมเดล kNN ด้วย cross-validation
    knn_scores = cross_val_score(knn_model, X_scaled, y, cv=kf, scoring='accuracy')
    knn_results.append(knn_scores.mean())

    # แสดงผลลัพธ์ในแต่ละรอบ
    print(f'Experiment {i:02d}: Naïve Bayes Accuracy: {nb_scores.mean():.2f}, kNN Accuracy: {knn_scores.mean():.2f}')

# แสดงผลลัพธ์เฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของ accuracy
print(f'\nNaïve Bayes Average Accuracy: {np.mean(nb_results):.2f} (+/- {np.std(nb_results) * 2:.2f})')
print(f'kNN Average Accuracy: {np.mean(knn_results):.2f} (+/- {np.std(knn_results) * 2:.2f})')