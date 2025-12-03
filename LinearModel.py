from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R²:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("\nWspółczynniki modelu:")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.4f}")

# Współczynniki mówią, jak zmienia się prewidywana cena domu gdy dana cecha rośnie o 1 jednostkę, przy założeniu stałości innych cech
# Jeśli współczynnik dodatni to cecha zwiększa cenę, jeśli ujemny to ją obniża
# Największa wartość współczynnika wskazuje, która cecha ma największy wpływ, w tym przypadku "AveBedrms"