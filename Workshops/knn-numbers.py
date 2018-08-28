# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
from sklearn import neighbors, datasets, model_selection, metrics
import pylab as pl

digits = datasets.load_digits() # se guardan los numeros en una variable
numImagenes = len(digits.images) # Numero de imagenes, len es un método que provee el tamaño del arreglo
print("Se tienen en total:", numImagenes, "imágenes")
y = digits.target # el método nos provee las etiquetas de las imágenes en un arreglo
x = digits.images.reshape((numImagenes, -1)) # se reducen las dimensiones

# y = [] # inicializamos un arreglo donde se almacenan las etiquetas de las imágenes que son 4 o 6
# x = [] # inicializamos un arreglo donde se almacenan las imágenes que corresponden a las etiquedas de 4 o 6
# for i in range(0, len(digits.target)):
    # if yy[i] == 4 or yy[i] == 6:
        # x.append(xx[i])
        # y.append(yy[i])

ts = 0.20 # porcentaje de datos utilizados para pruebas
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=ts) # obtenemos el x de prueba & y de prueba 20%

knn = neighbors.KNeighborsClassifier(n_neighbors=10) # Creamos una instancia de Neighbours Classifier
knn.fit(x_train, y_train) # Generamos el modelo con los datos y resultados de entrenamiento

y_pred = knn.predict(x_test) # realizamos una predicción para el x & y de prueba

class_report = metrics.classification_report(y_test, y_pred) # reporte de predicción

print(class_report)
