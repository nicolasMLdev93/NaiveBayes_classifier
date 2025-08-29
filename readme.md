📧 Naive Bayes Email Classifier

Este proyecto implementa un **clasificador de emails** que permite predecir si un correo electrónico es **Spam** o **No Spam** utilizando el algoritmo **Naive Bayes**.  

El modelo se entrena con un **dataset de emails etiquetados**, donde cada correo tiene asignada una etiqueta binaria:  
- `0` → No Spam  
- `1` → Spam  

El flujo del proyecto incluye:  
1. **Procesamiento del texto:** los correos se transforman en vectores numéricos utilizando `CountVectorizer`, que convierte las palabras en frecuencias para que el modelo pueda trabajar con ellas.  
2. **Entrenamiento del modelo:** se utiliza `Multinomial Naive Bayes`, un algoritmo muy eficiente para clasificación de texto, especialmente adecuado para problemas de detección de spam.  
3. **Predicciones:** el modelo puede predecir la probabilidad de que un correo sea spam o no, tanto sobre el conjunto de test real como sobre emails individuales generados de forma aleatoria.  
4. **Visualización de métricas:** se generan gráficos como la **curva Precision-Recall**, lo que permite evaluar el desempeño del modelo y la capacidad de separar correctamente spam de no spam.  

Este enfoque permite:  
- Detectar automáticamente correos no deseados en grandes volúmenes de emails.  
- Evaluar y ajustar el modelo mediante métricas y gráficos.  
- Generar emails falsos para probar la predicción del modelo de manera interactiva.  


🌐 Clonar el proyecto

git clone https://github.com/nicolasMLdev93/NaiveBayes_classifier.git

1 - Creo un enorno virtual y lo corro:

python -m venv venv 

venv\Scripts\activate

2 - Instalar dependencias:

pip install -r requirements.txt

3 - Entrenamiento del modelo y testeo:

python src/main.py

4 - Métricas:

python reports/metrics.py

-Curva Precision-Recall guardada como precision_recall_curve.png 📈

-Predicción de un email falso, mostrando si es spam y su probabilidad 💌

🧑 Mi perfil:

https://www.linkedin.com/in/nicol%C3%A1s-bauz%C3%A1-48a8a0244/ 👈 – Sígueme para ver mis proyectos de desarrollo y ML. 🚀



