import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Ensure all feature vectors have exactly 42 features
expected_feature_length = 42
filtered_data = [sample for sample in data_dict['data'] if len(sample) == expected_feature_length]
filtered_labels = [label for i, label in enumerate(data_dict['labels']) if len(data_dict['data'][i]) == expected_feature_length]

if not filtered_data:
    raise ValueError("No samples found with the expected feature length (42). Check your data preprocessing.")

# Convert to NumPy arrays
data = np.asarray(filtered_data, dtype=np.float32)
labels = np.asarray(filtered_labels)


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predict on the test set
y_predict = model.predict(x_test)

# Evaluate the model
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Print additional evaluation metrics
print(classification_report(y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("✅ Model trained and saved successfully!")
