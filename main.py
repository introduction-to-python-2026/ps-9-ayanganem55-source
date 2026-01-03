import pandas as pd
df = pd.read_csv('/content/parkinsons.csv')
display(df.head())

  X = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
  y = df['status']

  print("Input features (X) selected:")
  display(X.head())
  print("Output feature (y) selected:")
  display(y.head())
from sklearn.preprocessing import MinMaxScaler

X_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)

print("Scaled Input features (X_scaled) selected:")
display(pd.DataFrame(X_scaled, columns=X.columns).head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

print("Model selected:")
display(model)
 model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

if accuracy >= 0.8:
    print("Accuracy is sufficient (>= 0.8).")
else:
    print("Accuracy is below 0.8. Consider further tuning or a different model.")
