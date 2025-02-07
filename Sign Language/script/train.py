import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('text to audio/model3/dataset/isl_data1.csv')

# Separate features (landmarks) and labels
X = data.drop('label', axis=1)  # Features: landmark coordinates
y = data['label']  # Labels: gestures

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Encode the labels into numerical format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)

# Train a RandomForest model (or any other classifier)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model, scaler, and the label encoder
joblib.dump(clf, 'text to audio/model3/models/gesture_classifier.pkl')
joblib.dump(scaler, 'text to audio/model3/models/scaler.pkl')
joblib.dump(label_encoder, 'text to audio/model3/models/label_encoder.pkl')

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import joblib

# # Load the dataset
# data = pd.read_csv('text to audio/model3/dataset/isl_data.csv')

# # Separate features (landmarks for both hands) and labels
# X = data.drop('label', axis=1)  # Features: landmark coordinates for both hands
# y = data['label']  # Labels: gestures

# # Normalize the features
# scaler = StandardScaler()
# X_normalized = scaler.fit_transform(X)

# # Encode the labels into numerical format
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)

# # Train a RandomForest model (or any other classifier)
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train, y_train)

# # Evaluate the model
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy * 100:.2f}%')

# # Save the trained model, scaler, and the label encoder
# joblib.dump(clf, 'text to audio/model3/models/gesture_classifier.pkl')
# joblib.dump(scaler, 'text to audio/model3/models/scaler.pkl')
# joblib.dump(label_encoder, 'text to audio/model3/models/label_encoder.pkl')
