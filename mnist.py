# modelop.schema.0: input_schema.avsc
# modelop.schema.1: output_schema.avsc

import tensorflow as tf
import numpy as np


# modelop.init
def begin():
    global model
    # Loading model from trained artifact    
    model = tf.keras.models.load_model('./binaries/mnist.h5')


# modelop.score
def action(datum):
    
    # Compute 10 probabilities, 1 for each possible digit
    predicted_probs = model.predict(np.array([datum["array"]])).tolist()[0]
    
    # Add these probabilities to the output
    datum["predicted_probs"] = predicted_probs

    # Add the best possible matching digit to the output
    datum["score"] = np.argmax(predicted_probs)

    # Remove input array from output
    del datum["array"]
    
    yield datum
