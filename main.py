import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. MEMBUAT DATASET SIMULASI TRANSAKSI INVOICE
np.random.seed(42)
n_samples = 1000

data = {
    'Total_Amount': np.random.uniform(1000, 50000, n_samples),
    'Customer_Past_Late_Count': np.random.randint(0, 10, n_samples),
    'Credit_Term_Days': np.random.choice([15, 30, 45, 60], n_samples),
    'Discount_Given': np.random.uniform(0, 0.15, n_samples),
    'Payment_Method_Type': np.random.choice([1, 2, 3], n_samples)  # 1: Transfer, 2: Giro, 3: Cash
}

df = pd.DataFrame(data)

# Kriteria Target (1 = Terlambat/Late Payment, 0 = Tepat Waktu/On-Time)
df['Is_Late'] = np.where(
    (df['Customer_Past_Late_Count'] >= 3) | 
    ((df['Total_Amount'] > 30000) & (df['Credit_Term_Days'] <= 30)), 
    1, 0
)

# 2. PERSIAPAN DATA (DATA PREPARATION)
X = df.drop(columns=['Is_Late'])
y = df['Is_Late']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. PEMODELAN MACHINE LEARNING (RANDOM FOREST)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. EVALUASI MODEL
y_pred = model.predict(X_test)

print("="*55)
print(" HASIL EVALUASI PREDIKSI KETERLAMBATAN PEMBAYARAN INVOICE")
print("="*55)
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))