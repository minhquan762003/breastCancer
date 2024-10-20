import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Giả sử bạn có dữ liệu đầu vào X
X = [[2.5, 2.4],
     [0.5, 0.7],
     [2.2, 2.9],
     [1.9, 2.2],
     [3.1, 3.0],
     [2.3, 2.7],
     [2, 1.6],
     [1, 1.1],
     [1.5, 1.6],
     [1.1, 0.9]]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Áp dụng PCA
pca = PCA(n_components=1)  # Giảm chiều từ 2 xuống 1
X_pca = pca.fit_transform(X_scaled)

print("Dữ liệu gốc:", X)
print("Dữ liệu sau khi chuẩn hóa:", X_scaled)
print("Dữ liệu sau khi áp dụng PCA:", X_pca)
