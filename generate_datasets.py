import tensorflow as tf
import pandas as pd
import logging
import sys


logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

logger.info("Loading the MNIST dataset")
mnist = tf.keras.datasets.mnist

# Load training and testing data
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# x_train and x_test contain grayscale 28*28 images
# y_train and y_test contain the actual (ground truth) number (digit) that is in the picture

# Let's write the datasets into JSON-lines files

# We will assign an 'id' column to each record (image), as well as a 'label'. The pixel array is represented under 'array'
train_json = {'id':[], 'array':[], 'label':[]}
test_json = {'id':[], 'array':[], 'label':[]}

logger.info("Formatting training data")
cnt=0
for rec in x_train:
    train_json['id'].append(cnt)
    train_json['array'].append(rec.tolist())
    train_json['label'].append(y_train[cnt])
    cnt+=1

logger.info("Formatting testing data")
cnt=0
for rec in x_test:
    test_json['id'].append(cnt)
    test_json['array'].append(rec.tolist())
    test_json['label'].append(y_test[cnt])
    cnt+=1

train_records = pd.DataFrame(train_json)
test_records = pd.DataFrame(test_json)

logger.info("Writing training data to .data/train.json")
train_records.to_json("./data/train.json", orient="records", lines=True)

logger.info("Writing testing data to .data/test.json")
test_records.to_json("./data/test.json", orient="records", lines=True)

logger.info("DONE.")