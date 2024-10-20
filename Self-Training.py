from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 生成數據
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_unlabeled, y_train, _ = train_test_split(X, y, test_size=0.9, random_state=42)
X_unlabeled, X_test, _, y_test = train_test_split(X_unlabeled, y, test_size=0.5, random_state=42)

# 初始模型訓練
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 自訓練迭代
for _ in range(10):
    y_unlabeled_pred = model.predict(X_unlabeled)
    high_confidence_idx = np.where(model.predict_proba(X_unlabeled).max(axis=1) > 0.9)[0]
    X_train = np.vstack((X_train, X_unlabeled[high_confidence_idx]))
    y_train = np.hstack((y_train, y_unlabeled_pred[high_confidence_idx]))
    X_unlabeled = np.delete(X_unlabeled, high_confidence_idx, axis=0)
    model.fit(X_train, y_train)

# 評估模型
y_test_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))