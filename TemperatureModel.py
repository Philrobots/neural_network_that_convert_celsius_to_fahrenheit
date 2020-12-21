import numpy as np
import tensorflow as tf
import logging


def celsius_to_fahrenheit(celsius):
    return celsius * 1.8 + 32


logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)

print("Finished training the model 3500 times")

is_output_number = False

while not is_output_number:
    try:
        celsius_value = int(input("What value do you want to convert to fahrenheit ? : "))
    except ValueError:
        print("Not an integer ! Please try again")
        continue
    else:
        is_output_number = True

model_fahrenheit_value = model.predict([celsius_value])

print("My model say the fahrenheit value is : {}".format(model_fahrenheit_value))
print("The correct value is : {}".format(celsius_to_fahrenheit(celsius_value)))