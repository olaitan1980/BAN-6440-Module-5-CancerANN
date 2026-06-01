import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
data = pd.read_csv("breast_cancer_data.csv")

print("\n--- Missing Values ---")
print(data.isnull().sum())

# Drop unnecessary index column
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])

# Separate features and target
X = data.drop(columns=["y"])
y = data["y"]

# Encode labels: B = 0, M = 1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n--- SHAPES ---")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build ANN
model = Sequential([
    Dense(16, activation="relu", input_shape=(30,)),   # 30 input features
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")                     # Binary output
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n--- Training Model ---")
history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Evaluate on test data
print("\n--- Model Evaluation ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

import matplotlib.pyplot as plt

# Plot training accuracy and loss
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
# Save the trained ANN
model.save("cancer_ann_model.h5")
print("Model saved as cancer_ann_model.h5")

from sklearn.metrics import classification_report, confusion_matrix

# Predict on test data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to 0/1

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix ---")
print(cm)

# Classification Report
report = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
print("\n--- Classification Report ---")
print(report)

import os

# Create folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Accuracy plot
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("plots/training_accuracy.png")

# Loss plot
plt.figure()
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("plots/training_loss.png")

