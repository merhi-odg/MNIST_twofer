# MNIST_tf
A sample model to demonstrate **tensorflow** in ModelOp Center. The saved model was trained on the MNIST dataset.


## Assets
- `./binaries/mnist.h5` is the trained tensorflow model.
- `./binaries/tf_mnist_cp.h5` are the model weights, which could be used to load the model from script.
- `input_schema.avsc` and `output_schema.avsc` are AVRO-compliant json files that detail the input and output schema, respectively.
- The datasets used for **training** (`train.json`) and **testing** (`test.json`) are generated in the attached notebook, and uploaded to S3.

**_NOTE:_**  to generate the datasets **without** running the notebook, simply run the `generate_datasets.py` file (`python3 generate_datasets.py`).

## Scoring Jobs

### Sample Inputs

Choose `data/sample_input.json` as the input file. The scoring job requires a runtime with **python >= 3.8**.

### Schema Checking

Scoring Jobs can be run with any combination of input/output schema checking.

### Sample Output

Each output record is a dictionary with 4 keys: `id`, `label`, `predicted_probs` and `score`. Here's an example:
```json
{
    "id": 0,
    "label": 5,
    "predicted_probs": [
        5.110288345375365e-14,
        2.8476302912916474e-11,
        5.987289115015615e-13,
        0.0003364254371263087,
        1.5704866054112772e-21,
        0.9996635913848877,
        1.036646354694859e-15,
        9.677076306079113e-14,
        6.196956061363737e-14,
        7.691011716381979e-10
    ],
    "score": 5
}
```