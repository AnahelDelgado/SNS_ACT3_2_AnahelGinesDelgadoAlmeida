# 🧠 Naive Bayes con Datasets Iris y Penguins

Este repositorio contiene un ejercicio práctico para entrenar y evaluar distintos modelos de **Naive Bayes** utilizando los datasets **Iris** y **Penguins**.  
El objetivo principal es **comparar la precisión** de varios modelos y visualizar sus **matrices de confusión**.  

---

## 📦 Contenido

- **Funciones principales**
  - `evaluar_y_graficar(modelo, Xtrain, Xtest, ytrain, ytest, titulo)`  
    🔹 Entrena un modelo de clasificación  
    🔹 Calcula la precisión  
    🔹 Genera la matriz de confusión como gráfico  

- **Datasets utilizados**
  1. **Iris** 🌸  
     - Variables: medidas de sépalos y pétalos  
     - Clases: `setosa`, `versicolor`, `virginica`  
  2. **Penguins** 🐧  
     - Variables: características físicas y categóricas de pingüinos  
     - Clases: diferentes especies de pingüinos  

- **Modelos Naive Bayes**
  1. `GaussianNB` – Datos continuos  
  2. `MultinomialNB` – Datos discretos y no negativos  
  3. `ComplementNB` – Variante del MultinomialNB, útil con clases desbalanceadas  
  4. `BernoulliNB` – Datos binarios  
  5. `CategoricalNB` – Variables categóricas discretizadas  

---

## 🚀 Ejecución del ejercicio

1. Cargar y preparar los datasets (limpieza, conversión de categóricas a dummies, división en train/test)  
2. Aplicar los distintos modelos Naive Bayes a cada dataset  
3. Evaluar la **precisión** y visualizar la **matriz de confusión**  
4. Preprocesamiento especial para algunos modelos:  
   - `BernoulliNB`: binarización según la mediana  
   - `CategoricalNB`: discretización en bins  

---

## 📝 Ejemplo de uso

```python
from sklearn.naive_bayes import GaussianNB

acc = evaluar_y_graficar(
    GaussianNB(), 
    Xtrain, Xtest, 
    ytrain, ytest, 
    "GaussianNB"
)
print(f"Precisión: {acc:.2f}")
