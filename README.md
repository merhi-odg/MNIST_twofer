# MNIST_tf
A sample model to demonstrate **tensorflow** and **sklearn** in ModelOp Center. The saved models were trained on the MNIST dataset.


## Assets
- `./binaries/mnist.h5` is the trained tensorflow model.
- `./binaries/tf_mnist_cp.h5` are the model weights, which could be used to load the model from script.
- `./binaries/sklearn_mnist.pkl` is the trained sklearn model.
- `input_schema.avsc` and `output_schema.avsc` are AVRO-compliant json files that detail the input and output schema, respectively.
- The datasets used for **training** (`train.json`) and **testing** (`test.json`) are generated in the attached notebook, and uploaded to S3.

**_NOTE:_**  to generate the datasets **without** running the notebook, simply run the `generate_datasets.py` file (`python3 generate_datasets.py`).

## Scoring Jobs

### Sample Inputs

Choose `data/sample_input.json` as the input file. The scoring job requires a runtime with **python >= 3.8**.

### Schema Checking

Scoring Jobs can be run with any combination of input/output schema checking.

### Sample Output

Each output record is a dictionary with 6 keys: `id`, `label`, `max_prob_tf`, `max_prob_sklearn`, `score_tf` and `score_sklearn`. Here's an example:
```json
{
    "id": 0,
    "label": 7,
    "max_prob_tf": 0.9999997615814209,
    "max_prob_sklearn": 0.9948027195804962,
    "score_tf": 7,
    "score_sklearn": 7
}
```