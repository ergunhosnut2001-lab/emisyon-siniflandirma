import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import os

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Excel dosyasının bulunduğu tam yolu alıyoruz
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.xlsx")
BG_IMAGE_PATH = os.path.join(BASE_DIR, "bg.png")

# Sınıf Eşleştirmesi
CLASS_MAPPING = {
    0: "Tarım Sektörü",
    1: "Sanayi Sektörü",
    2: "Enerji Üretimi",
    3: "Ulaşım"
}

model_loaded = False
error_msg = ""
model = None

try:
    df = pd.read_excel(DATASET_PATH, decimal=",")
    X = df[["CO2", "N2O", "CH4"]]
    y = df["class"]
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ])
    model.fit(X, y)
    model_loaded = True
except Exception as e:
    error_msg = str(e)

def predict():
    if not model_loaded:
        messagebox.showerror("Hata", f"Model yüklenemedi. Lütfen 'dataset.xlsx' dosyasını kontrol edin.\nHata Detayı: {error_msg}")
        return
        
    try:
        co2 = float(entry_co2.get().replace(',', '.'))
        n2o = float(entry_n2o.get().replace(',', '.'))
        ch4 = float(entry_ch4.get().replace(',', '.'))
        
        prediction = model.predict([[co2, n2o, ch4]])[0]
        result_text = CLASS_MAPPING.get(prediction, f"Bilinmeyen Sınıf ({prediction})")
        lbl_result.config(text=f"Tahmin: {result_text}", fg="#ffffff", bg="#1b5e20")
    except ValueError:
        messagebox.showwarning("Geçersiz Veri", "Lütfen tüm alanlara geçerli sayısal değerler girin.")

# -----------------------------
# Arayüz (GUI) Tasarımı
# -----------------------------
root = tk.Tk()
root.title("Yapay Zeka Emisyon Tahmini")
root.geometry("600x600")
root.resizable(False, False)

# Arka Plan Resmi
if HAS_PIL and os.path.exists(BG_IMAGE_PATH):
    try:
        pil_img = Image.open(BG_IMAGE_PATH)
        pil_img = pil_img.resize((600, 600), Image.Resampling.LANCZOS)
        bg_img = ImageTk.PhotoImage(pil_img)
        canvas = tk.Canvas(root, width=600, height=600, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, image=bg_img, anchor="nw")
        canvas.bg_img = bg_img  # Garbage collection'ı önlemek için referans tutuyoruz
    except Exception as e:
        root.configure(bg="#2E7D32")
        canvas = tk.Canvas(root, width=600, height=600, bg="#2E7D32", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
else:
    root.configure(bg="#2E7D32")
    canvas = tk.Canvas(root, width=600, height=600, bg="#2E7D32", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

# Orta Panel
panel = tk.Frame(root, bg="#ffffff", bd=0, padx=40, pady=40)
panel.configure(highlightbackground="#cccccc", highlightcolor="#cccccc", highlightthickness=1)
panel.place(relx=0.5, rely=0.5, anchor="center")

tk.Label(panel, text="Emisyon Değerleri", font=("Segoe UI", 20, "bold"), bg="#ffffff", fg="#1b5e20").pack(pady=(0, 25))

frame = tk.Frame(panel, bg="#ffffff")
frame.pack()

font_label = ("Segoe UI", 12, "bold")
font_entry = ("Segoe UI", 12)

# CO2
tk.Label(frame, text="CO2 (Karbondioksit):", font=font_label, bg="#ffffff", fg="#333333").grid(row=0, column=0, padx=10, pady=12, sticky="e")
entry_co2 = tk.Entry(frame, font=font_entry, width=15, justify="center", bg="#f1f8e9", relief="solid", bd=1)
entry_co2.grid(row=0, column=1, padx=10, pady=12)

# N2O
tk.Label(frame, text="N2O (Azot Oksit):", font=font_label, bg="#ffffff", fg="#333333").grid(row=1, column=0, padx=10, pady=12, sticky="e")
entry_n2o = tk.Entry(frame, font=font_entry, width=15, justify="center", bg="#f1f8e9", relief="solid", bd=1)
entry_n2o.grid(row=1, column=1, padx=10, pady=12)

# CH4
tk.Label(frame, text="CH4 (Metan):", font=font_label, bg="#ffffff", fg="#333333").grid(row=2, column=0, padx=10, pady=12, sticky="e")
entry_ch4 = tk.Entry(frame, font=font_entry, width=15, justify="center", bg="#f1f8e9", relief="solid", bd=1)
entry_ch4.grid(row=2, column=1, padx=10, pady=12)

# Tahmin Butonu
btn_predict = tk.Button(panel, text="Tahmin Et", font=("Segoe UI", 14, "bold"), bg="#1b5e20", fg="white", 
                        activebackground="#2e7d32", activeforeground="white", width=18, cursor="hand2", relief="flat", command=predict)
btn_predict.pack(pady=30)

# Sonuç Gösterim Etiketi
lbl_result = tk.Label(panel, text="Henüz tahmin yapılmadı", font=("Segoe UI", 14, "bold"), bg="#ffffff", fg="#757575", padx=15, pady=8)
lbl_result.pack(pady=5)

if not model_loaded:
    lbl_result.config(text="Model Yüklenemedi!", fg="red")

root.mainloop()

