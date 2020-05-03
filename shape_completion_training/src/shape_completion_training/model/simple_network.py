

'''

Very simple neural network created by bsaund to practice coding in 
Tensorflow 2.0 (instead of 1.0)

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import IPython


class SimpleNetwork:
    def __init__(self):
        self.build_example_model()


    def simple_pass(self):
        W = tf.Variable(tf.ones(shape=(2,2)), name="W")
        b = tf.Variable(tf.zeros(shape=(2)), name="b")

        @tf.function
        def forward(x):
            return W * x + b

        out_a = forward([1,0])
        print(out_a)
    
    #     self.W = tf.Variable(tf.ones(shape=(2,2)), name="W")
    #     self.b = tf.Variable(tf.ones(shape=(2)), name="b")

    # @tf.function
    # def forward(self, x):
    #     return self.W * x + self.b
    

    def build_example_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.04),
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def forward_model(self):
        train_data = tf.ones(shape=(1, 28, 28, 1))
        test_data = tf.ones(shape=(1, 28, 28, 1))
        
        train_out = self.model(train_data, training=True)
        print("Train out")
        print(train_out)


        test_out = self.model(test_data, training=False)
        print()
        print("Test out")
        print(test_out)

    def mnist_train_example(self):
        # train, test = tf.keras.datasets.fashion_mnist.load_data()
        train, test = tf.keras.datasets.mnist.load_data()

        images, labels = train
        images = images/255.0
        images = np.expand_dims(images, axis=3)
        labels = labels.astype(np.int32)

        test_size = 20000
        
        # IPython.embed()        
        fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
        # fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)
        fmnist_train_ds = fmnist_train_ds.shuffle(5000)
        fmnist_test_ds =  fmnist_train_ds.take(test_size).batch(32)
        fmnist_train_ds = fmnist_train_ds.skip(test_size).batch(32)



        
        model = tf.keras.Sequential([
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(10, 3, input_shape=(28, 28,1)),
            tf.keras.layers.MaxPool2D((2,2)),
            # tf.keras.layers.Conv2D(100, 3),
            # tf.keras.layers.MaxPool2D((2,2)),
            # tf.keras.layers.Conv2D(100, 3),
            # tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                      metrics=['accuracy'])

        # model.evaluate()
        model.fit(fmnist_train_ds, epochs=10)
        result = model.predict(fmnist_train_ds)
        print(result.shape)
        model.evaluate(fmnist_test_ds)




if __name__ == "__main__":
    print("hi")
    sn = SimpleNetwork()
    # sn.simple_pass()
    # sn.forward_model()
    sn.mnist_train_example()
