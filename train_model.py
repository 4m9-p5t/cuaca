import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

data = {
    'suhu': [28, 30, 27, 22, 24, 35, 31],
    'kelembapan': [80, 60, 90, 95, 85, 40, 55],
    'tekanan': [1008, 1012, 1005, 1002, 1007, 1010, 1009],
    'hujan': [1, 0, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

X = df[['suhu', 'kelembapan', 'tekanan']]
y = df['hujan']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Buat model dan latih
model = RandomForestClassifier()
model.fit(X_scaled, y)

os.makedirs('d:/dcode 5/Cuaca/model', exist_ok=True)

joblib.dump(model, 'd:/dcode 5/Cuaca/model/model.pkl')
joblib.dump(scaler, 'd:/dcode 5/Cuaca/model/scaler.pkl')

print("✅ Model dan Scaler berhasil disimpan.")
