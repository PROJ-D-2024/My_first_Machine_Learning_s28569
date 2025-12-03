# Machine Learning Mini-Project  
### Porównanie trzech modeli: Linear Regression, Random Forest, Naive Bayes

Ten projekt przedstawia implementację i analizę trzech różnych modeli uczenia maszynowego, zbudowanych na trzech różnych typach danych: regresyjnych, klasyfikacyjnych oraz tekstowych.

W projekcie wykorzystano trzy klasyczne zbiory danych z biblioteki **scikit-learn**:

- **California Housing** – regresja
- **Iris Dataset** – klasyfikacja tabularna
- **20 Newsgroups** – klasyfikacja tekstu
# Linear Regression
```
- R²: 0.5757877060324521
- MSE: 0.5558915986952425

Współczynniki modelu:
- MedInc: 0.4487
- HouseAge: 0.0097
- AveRooms: -0.1233
- AveBedrms: 0.7831
- Population: -0.0000
- AveOccup: -0.0035
- Latitude: -0.4198
- Longitude: -0.4337
```

Współczynniki mówią, jak zmienia się prewidywana cena domu gdy dana cecha rośnie o 1 jednostkę, przy założeniu stałości innych cech. Jeśli współczynnik dodatni to cecha zwiększa cenę, jeśli ujemny to ją obniża. Największa wartość współczynnika wskazuje, która cecha ma największy wpływ, w tym przypadku "AveBedrms".
# Tree-Based Model

```Accuracy: 1.0

              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11
    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```
# Naive Bayes
```Accuracy: 0.9780405405405406
              precision    recall  f1-score   support

           0       0.98      0.96      0.97       195
           1       0.98      0.99      0.99       191
           2       0.97      0.99      0.98       206

    accuracy                           0.98       592
   macro avg       0.98      0.98      0.98       592
weighted avg       0.98      0.98      0.98       592
```
# Podsumowanie
Model liniowy okazał się najszybszy do trenowania, ponieważ operuje na prostych obliczeniach macierzowych. Random Forest działał znacznie wolniej, ale zapewnił najlepszą dokładność, ponieważ bardzo dobrze radzi sobie z danymi tabelarycznymi o nieliniowych zależnościach. Naiwny Bayes był również bardzo szybki — szczególnie przy dużych zbiorach tekstowych — i osiągnął dobrą, choć nie najlepszą dokładność. Model liniowy był najłatwiejszy do interpretacji, ponieważ współczynniki jasno pokazują wpływ każdej cechy na wynik. Z kolei model drzewiasty jest trudniejszy do wyjaśnienia, ale jego działanie można wizualizować. Najtrudniejszym zbiorem danych był 20 Newsgroups, ponieważ tekst wymaga wektoryzacji i jest bardziej złożony niż dane liczbowe. 
Do prostych danych tabelarycznych, gdzie ważna jest skuteczność, wybrałbym Random Forest.
Do szybkiej klasyfikacji tekstu lub filtrowania spamu na start wybrałbym Naive Bayes ze względu na szybkość i prostotę.
Ze względu na przejrzystość, wybrałbym Regresję Liniową lub Drzewo Decyzyjne.