
import mnist_reader
import preprocessing
import network

X_train, y_train = mnist_reader.load_mnist('data', kind='train')
X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')

preprocessing.batch_data(X_train, y_train)


