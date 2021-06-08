# Q4_graded
# Do not change the above line.

# This cell is for your imports.

from keras.layers import *
from keras.optimizers import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


class MLP(object):
    def __init__(self, num_classes, X_train, Y_train, X_test, Y_test):
        self.model = None
        self.num_classes = num_classes
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.history = None

    def create_MLP(self, inputs):
        # Create the model
        self.model = Sequential()
        self.model.add(Dense(100, input_shape=inputs, activation='relu'))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def reshape_dataset(self, feature_vector_length=784):
        # Reshape the data - MLPs do not understand such things as '2D'.
        # Reshape to 28 x 28 pixels = 784 features
        self.X_train = self.X_train.reshape(self.X_train.shape[0], feature_vector_length)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], feature_vector_length)

        # Convert into greyscale
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255

        # Convert target classes to categorical ones
        self.Y_test = to_categorical(self.Y_test, self.num_classes)
        self.Y_train = to_categorical(self.Y_train, self.num_classes)

    def start_training(self, epochs=10, batch_size=250, verbose=1, validation_split=0.2):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = self.model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                       validation_split=validation_split)

    def evaluate(self, verbose=1):
        test_results = self.model.evaluate(self.X_test, self.Y_test, verbose=verbose)
        print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')


feature_vector_length = 784
num_classes = 10

# Set the input shape
input_shape = (feature_vector_length,)
print(f'Feature shape: {input_shape}')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

mlp = MLP(num_classes, x_train, y_train, x_test, y_test)
mlp.create_MLP(input_shape)
mlp.reshape_dataset()
mlp.start_training()
mlp.evaluate()

#plot
plt.plot(mlp.history.history['accuracy'])
plt.plot(mlp.history.history['val_accuracy'])
plt.title('model accuracy') 
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(mlp.history.history['loss'])
plt.plot(mlp.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Q4_graded
# Do not change the above line.

# This cell is for your codes.



# Q4_graded
# Do not change the above line.

# This cell is for your codes.

