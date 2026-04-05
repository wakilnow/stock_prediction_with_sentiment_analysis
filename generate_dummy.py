import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

os.makedirs("data/processed", exist_ok=True)
# X shape: (samples, seq_length, num_features)
X_train = np.random.rand(100, 30, 4).astype(np.float32)
X_test = np.random.rand(20, 30, 4).astype(np.float32)

# y shape: (samples,)
y_train = np.random.rand(100).astype(np.float32)
y_test = np.random.rand(20).astype(np.float32)

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)

# Mock scaler
scaler = MinMaxScaler()
scaler.fit(np.random.rand(100, 1) * 100) # Dummy fit
joblib.dump(scaler, "data/processed/scaler.save")

print("Created dummy data in data/processed/")
