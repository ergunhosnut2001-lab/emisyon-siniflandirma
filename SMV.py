
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# -----------------------------
# Excel oku
# -----------------------------
# Not: dataset.xlsx dosyasının kodla aynı klasörde olduğundan emin olun.
try:
    df = pd.read_excel("dataset.xlsx", decimal=",")
except FileNotFoundError:
    print("Hata: 'dataset.xlsx' dosyası bulunamadı!")
    exit()

X = df[["CO2", "N2O", "CH4"]]
y = df["class"]

print("Sınıf dağılımı:")
print(y.value_counts())

# -----------------------------
# 5-Fold CV Ayarları
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}

results = []

# KNN grafik verileri için listeler
k_values = range(2, 13)
knn_metrics = {"acc": [], "pre": [], "rec": [], "f1": []}

# -----------------------------
# KNN (k=2–12) Döngüsü
# -----------------------------
for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring)

    # Ortalama metrikler
    acc = cv_res["test_accuracy"].mean()
    pre = cv_res["test_precision"].mean()
    rec = cv_res["test_recall"].mean()
    f1 = cv_res["test_f1"].mean()

    results.append({
        "Model": "KNN",
        "Parametre": f"k={k}",
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1-Score": f1
    })

    knn_metrics["acc"].append(acc)
    knn_metrics["pre"].append(pre)
    knn_metrics["rec"].append(rec)
    knn_metrics["f1"].append(f1)

# -----------------------------
# SVM ve Naive Bayes
# -----------------------------
# SVM
svm_pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))])
svm_cv = cross_validate(svm_pipe, X, y, cv=cv, scoring=scoring)
results.append({
    "Model": "SVM", "Parametre": "RBF Kernel",
    "Accuracy": svm_cv["test_accuracy"].mean(),
    "Precision": svm_cv["test_precision"].mean(),
    "Recall": svm_cv["test_recall"].mean(),
    "F1-Score": svm_cv["test_f1"].mean()
})

# Naive Bayes
nb_cv = cross_validate(GaussianNB(), X, y, cv=cv, scoring=scoring)
results.append({
    "Model": "Naive Bayes", "Parametre": "-",
    "Accuracy": nb_cv["test_accuracy"].mean(),
    "Precision": nb_cv["test_precision"].mean(),
    "Recall": nb_cv["test_recall"].mean(),
    "F1-Score": nb_cv["test_f1"].mean()
})

# -----------------------------
# Excel'e Kaydet
# -----------------------------
pd.DataFrame(results).to_excel("siniflandirma_sonuclari.xlsx", index=False)
print("Sonuçlar 'siniflandirma_sonuclari.xlsx' dosyasına kaydedildi.")

# -----------------------------
# KNN Grafik Çizimi
# -----------------------------

plt.figure(figsize=(10, 6))
plt.plot(k_values, knn_metrics["acc"], marker="o", label="Accuracy")
plt.plot(k_values, knn_metrics["pre"], marker="s", label="Precision")
plt.plot(k_values, knn_metrics["rec"], marker="^", label="Recall")
plt.plot(k_values, knn_metrics["f1"], marker="x", label="F1-Score")

plt.title("KNN Modeli: K Değerine Göre Performans Değişimi")
plt.xlabel("Komşu Sayısı (k)")
plt.ylabel("Skor")
plt.xticks(k_values)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()