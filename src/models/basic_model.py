from models.model import Model
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.layers import Dropout

class BasicModel(Model):
    def __init__(self, input_shape, categories_count, learning_rate=0.001, conv_layers=2, dense_units=84, dropout_rate=0.0):
        self.learning_rate = learning_rate
        self.conv_layers = conv_layers
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        super().__init__(input_shape, categories_count)

    def _define_model(self, input_shape, categories_count):
        self.model = Sequential()
        self.model.add(layers.Rescaling(1./255, input_shape=input_shape))
        
        for _ in range(self.conv_layers):
            self.model.add(layers.Conv2D(16, (3, 3), activation='relu'))
            self.model.add(layers.MaxPooling2D((4, 4)))
            if self.dropout_rate > 0.0:
                self.model.add(Dropout(self.dropout_rate))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.dense_units, activation='relu'))
        
        if self.dropout_rate > 0.0:
            self.model.add(Dropout(self.dropout_rate))
        
        self.model.add(layers.Dense(categories_count, activation='softmax'))

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )