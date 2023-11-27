from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test_data, y_train, y_test_data = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

x_train = x_train.astype('float32') / 255.0
x_test_data = x_test_data.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(x_train.shape[0], -1)
x_test_data = x_test_data.reshape(x_test_data.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test_data = scaler.transform(x_test_data)
x_test = scaler.transform(x_test)

n_components = 100
pca = PCA(n_components=n_components)
x_train_pca = pca.fit_transform(x_train)
x_test_data_pca = pca.transform(x_test_data)
x_test_pca = pca.transform(x_test)

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(x_train_pca, y_train)

y_pred = knn_classifier.predict(x_test_pca)
accuracy = metrics.accuracy_score(y_test, y_pred)

print(f'Точність моделі: {accuracy}')
