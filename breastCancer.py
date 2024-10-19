import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tkinter as tk
from tkinter import messagebox

#Tiền xử lý
df = pd.read_csv('BRCA.csv')
df = df.drop(columns=['Patient_ID'])
df = df.drop(columns=['ER status'])
df = df.drop(columns=['PR status'])



# Khởi tạo LabelEncoder

# Xóa các dòng dữ liệu thiếu
df = df.dropna(axis=0)
label_encoder = LabelEncoder()
#Chuyển dữ liệu chữ thành số
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Tumour_Stage'] = label_encoder.fit_transform(df['Tumour_Stage'])
df['Histology'] = label_encoder.fit_transform(df['Histology'])
df['HER2 status'] = label_encoder.fit_transform(df['HER2 status'])
df['Surgery_type'] = label_encoder.fit_transform(df['Surgery_type'])
df['Date_of_Surgery'] = label_encoder.fit_transform(df['Date_of_Surgery'])
df['Date_of_Last_Visit'] = label_encoder.fit_transform(df['Date_of_Last_Visit'])
df['Patient_Status'] = label_encoder.fit_transform(df['Patient_Status'])



#Xuất dữ liệu đã xử lý vào file csv khác
df.to_csv('output.csv', index=False)
X = df.drop('Patient_Status', axis=1)
y = df['Patient_Status']

    # Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

    # Giảm chiều dữ liệu bằng PCA
pca = PCA(n_components=10)
X = pca.fit_transform(X)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Cân bằng dữ liệu bằng SMOTE
oversample = SMOTE(random_state=0)
X_train, y_train = oversample.fit_resample(X_train, y_train)

    # Định nghĩa các siêu tham số cần tìm kiếm
param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
}

    # Sử dụng GridSearchCV với cross-validation để tìm kiếm các siêu tham số tốt nhất
grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%')

    # Huấn luyện mô hình với các tham số tốt nhất
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
y_pred = best_model.predict(X_test)
# print(X_test)
    # Tính toán các độ đo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f'Độ chính xác: {accuracy * 100:.2f}%')
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

def predict():
        try:
            input_data = {
                'Age': [int(entry_age.get())],
                'Gender': [int(entry_gender.get())],
                'Protein1': [float(entry_protein1.get())],
                'Protein2': [float(entry_protein2.get())],
                'Protein3': [float(entry_protein3.get())],
                'Protein4': [float(entry_protein4.get())],
                'Tumour_Stage': [int(entry_tumour_stage.get())],
                'Histology': [int(entry_histology.get())],
                'HER2 status': [int(entry_her2_status.get())],
                'Surgery_type': [int(entry_surgery_type.get())],
                'Date_of_Surgery': [int(entry_date_of_surgery.get())],
                'Date_of_Last_Visit': [int(entry_date_of_last_visit.get())]
            }
            input_df = pd.DataFrame(input_data)
            
            # Chuẩn hóa dữ liệu
            input_scaled = scaler.transform(input_df)

            # Giảm chiều dữ liệu
            input_pca = pca.transform(input_scaled)

            # Dự đoán
            prediction = best_model.predict(input_pca)
            prediction_label = label_encoder.inverse_transform(prediction)  # Chuyển đổi nhãn dự đoán sang dạng chữ

            messagebox.showinfo("Dự đoán", f'Trạng thái bệnh nhân: {prediction_label[0]}')
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))
root = tk.Tk()
root.title("Dự đoán tình trạng bệnh nhân")

tk.Label(root, text="Age").grid(row=0)
tk.Label(root, text="Gender").grid(row=1)
tk.Label(root, text="Protein 1").grid(row=2)
tk.Label(root, text="Protein 2").grid(row=3)
tk.Label(root, text="Protein 3").grid(row=4)
tk.Label(root, text="Protein 4").grid(row=5)
tk.Label(root, text="Tumour Stage").grid(row=6)
tk.Label(root, text="Histology").grid(row=7)
tk.Label(root, text="HER2 status").grid(row=10)
tk.Label(root, text="Surgery type").grid(row=11)
tk.Label(root, text="Date of Surgery").grid(row=12)
tk.Label(root, text="Date of Last Visit").grid(row=13)

#Tạo các label
entry_age = tk.Entry(root)
entry_gender = tk.Entry(root)
entry_protein1 = tk.Entry(root)
entry_protein2 = tk.Entry(root)
entry_protein3 = tk.Entry(root)
entry_protein4 = tk.Entry(root)
entry_tumour_stage = tk.Entry(root)
entry_histology = tk.Entry(root)
entry_her2_status = tk.Entry(root)
entry_surgery_type = tk.Entry(root)
entry_date_of_surgery = tk.Entry(root)
entry_date_of_last_visit = tk.Entry(root)


#Tạo các trường input
entry_age.grid(row=0, column=1)
entry_gender.grid(row=1, column=1)
entry_protein1.grid(row=2, column=1)
entry_protein2.grid(row=3, column=1)
entry_protein3.grid(row=4, column=1)
entry_protein4.grid(row=5, column=1)
entry_tumour_stage.grid(row=6, column=1)
entry_histology.grid(row=7, column=1)
entry_her2_status.grid(row=10, column=1)
entry_surgery_type.grid(row=11, column=1)
entry_date_of_surgery.grid(row=12, column=1)
entry_date_of_last_visit.grid(row=13, column=1)

tk.Button(root, text='Dự đoán', command=predict).grid(row=14, column=1, pady=10)
root.mainloop()




