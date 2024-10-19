import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

try:
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

    # Xuất dữ liệu đã xử lý vào file csv khác
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

    # Tính toán các độ đo
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    print(f'Độ chính xác: {accuracy * 100:.2f}%')
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(report)

except Exception as e:
    print(f'Có lỗi xảy ra: {str(e)}')
