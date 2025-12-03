from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load text data
categories = ["sci.space", "comp.graphics", "rec.sport.hockey"]
data = fetch_20newsgroups(
    subset="all",
    categories=categories,
    download_if_missing=True
)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data.data)
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Print sample predictions
for i in range(5):
    print(f"\nTEXT: {data.data[i][:150]}...")
    print("Predicted label:", model.predict(X[i])[0])
