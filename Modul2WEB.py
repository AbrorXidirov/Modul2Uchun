import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Ma'lumotlarni yuklash
url = "https://raw.githubusercontent.com/AbrorXidirov/Modul2Uchun/refs/heads/main/diabetes_dataset.csv"
data = pd.read_csv(url)

# X (kirish o'zgaruvchilari) va y (natija) ni aniqlash
X = data[['age', 'blood_glucose_level', 'bmi', 'hypertension', 'heart_disease',
          'race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other']]
y = data['diabetes']  # Maqsadli o'zgaruvchi (kasallik holati)

# Datasetni 80% train va 20% test uchun ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)

# Ma'lumotlarni skalalash (standartlashtirish)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest modelini yaratish
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Modelni sinab ko'rish
y_pred = rf.predict(X_test)

# Modelning aniqligini baholash
def evaluation(y_test, y_pred):
    print(f"Model Accuracy: {metrics.accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Classification Report:\n {metrics.classification_report(y_test, y_pred)}")
    cm = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label", fontsize=15)
    plt.ylabel("Actual Label", fontsize=15)
    plt.show()

# Modelni baholash
evaluation(y_test, y_pred)

# Modelni faylga saqlash
with open('RandomForestasl.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Agar modelni Streamlit orqali ishlatish kerak bo'lsa:
import streamlit as st

# Streamlit interfeysi
st.title("Bemorni tekshirish:")
#'race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other'
# Kirish qiymatlarini olish
Age = st.number_input("Yosh: ", min_value=0, max_value=120)
Glukoza = st.number_input("Glukoza miqdori: ", format="%.1f", min_value=0.0)
BMI = st.number_input("BMI: ", format="%.1f", min_value=0.0)
Hypertension = st.selectbox("Gipertenziya mavjudmi?", options=[0, 1])
HeartDisease = st.selectbox("Yurak kasalligi mavjudmi?", options=[0, 1])
AfricanAmerican = st.selectbox("Irq: Afro-amerikalik", options=[0, 1])
Asian = st.selectbox("Irq: Osiyolik", options=[0, 1])
Caucasian = st.selectbox("Irq: Kavkazlik", options=[0, 1])
Hispanic = st.selectbox("Irq: Ispan tilida so'zlashuvchi", options=[0, 1])
Other = st.selectbox("Irq: Boshqa", options=[0, 1])

# Bashorat qilish uchun tugma
if st.button("Bashorat qilish"):
    features = np.array([[Age, Glukoza, BMI, Hypertension, HeartDisease, AfricanAmerican, Asian, Caucasian, Hispanic, Other ]])
    features = scaler.transform(features) 

    # Modelni yuklash
    try:
        with open('RandomForestasl.pkl', 'rb') as file:
            decision_tree_model = pickle.load(file)
    except Exception as e:
        st.error(f"Modelni yuklashda xato: {e}")

    # Bashorat qilish
    prediction = decision_tree_model.predict(features)

    if prediction[0] == 0:
        st.success("Bashorat: Sizda kasallik aniqlandi.")
    else:
        st.success("Bashorat: Sizning holatingiz yaxshi, kasallik aniqlanmadi.")
