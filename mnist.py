# modelop.schema.0: input_schema.avsc
# modelop.schema.1: output_schema.avsc

import tensorflow as tf
import joblib
import numpy as np


# modelop.init
def begin() -> None:
    
    global model_tf, model_sklearn
    # Loading model from trained artifact
    model_tf = tf.keras.models.load_model("./binaries/mnist.h5")
    model_sklearn = joblib.load(open("./binaries/sklearn_mnist.pkl", "rb"))


# modelop.score
def action(datum: dict) -> dict:

    # Compute 10 probabilities, 1 for each possible digit
    predicted_probs_tf = model_tf.predict(np.array([datum["array"]])).tolist()[0]
    predicted_probs_sklearn = model_sklearn.predict_proba(
        np.array(datum["array"]).ravel().reshape(1, -1)
    ).tolist()[0]

    # Add the max probability to output
    datum["max_prob_tf"] = np.max(predicted_probs_tf)
    datum["max_prob_sklearn"] = np.max(predicted_probs_sklearn)

    # Add the best possible matching digit to the output
    datum["score_tf"] = np.argmax(predicted_probs_tf)
    datum["score_sklearn"] = np.argmax(predicted_probs_sklearn)

    # Remove input array from output
    del datum["array"]

    return datum
