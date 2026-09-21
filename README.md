# NeuroJava

A feed-forward neural network implemented from first principles in Java 17 — no TensorFlow,
no PyTorch, no DL4J, no Weka — with a JavaFX interface that shows the network learning.

The matrix maths, weight initialisation, forward propagation, loss, backpropagation and
gradient descent are all written by hand. The UI exists to make that machinery visible:
architecture, weights, neuron activations, the loss curve and the decision boundary.

## Requirements

- JDK 17 or newer
- Maven (`brew install maven`)

JavaFX is pulled in by Maven; there is nothing to install separately.

## Run

```
mvn clean javafx:run
```

## Verify the maths

Each engine package ships a runnable self-check. They compare hand-derived gradients against
central finite differences, which is what proves backpropagation is actually correct:

```
mvn -q compile
java -cp target/classes com.neurojava.activation.MathCheck
java -cp target/classes com.neurojava.nn.GradientCheck
java -cp target/classes com.neurojava.data.DataCheck
```

`MathCheck` and `GradientCheck` are the ones that matter: they compare every hand-derived
gradient against a central finite difference. If backpropagation were wrong, they would fail.

## Reproduce the experiments

```
java -cp target/classes com.neurojava.experiments.ExperimentRunner
```

Sweeps learning rate, hidden width, activation and epoch budget, averaged over several seeds.
Every number is measured at run time — nothing in the report is hard-coded.

### One result worth knowing before the demo

XOR with sigmoid needs a much larger learning rate than the usual textbook 0.01:

| Setting | Converged (20 seeds) |
|---------|----------------------|
| Sigmoid, lr 0.01, 1000 epochs | never |
| Sigmoid, lr 0.1, 1000 epochs | 2 / 20 |
| ReLU, lr 0.1, 2000 epochs | 16 / 20 (dead units) |
| **Sigmoid, lr 0.5, 2000 epochs** | **20 / 20** |

The app defaults to the last row, so XOR converges every time you press Train.

## Demo script

1. Pick **XOR**, hidden layer `4`, activation `Sigmoid`, learning rate `0.1`, epochs `1000`.
2. **Initialize** — the network starts from random weights and predicts nothing useful.
3. **Train** — watch the loss curve fall and the connection thicknesses change.
4. **Predict** `0, 1` — the output should sit close to 1.
5. Switch to the **decision boundary** view to see the non-linear split XOR requires.
6. Repeat with **Iris** for multiclass classification with a softmax head.

## Layout

```
core/        Matrix — immutable, row-major, one row per sample
activation/  ReLU, Sigmoid, Softmax (+ their derivatives)
loss/        Mean squared error, binary and categorical cross entropy
nn/          Layer, DenseLayer, NeuralNetwork
training/    Trainer, TrainingConfig, Metrics, Accuracy
data/        Dataset, XorDataset, IrisDataset, Split
viz/         JavaFX dashboard, network view, loss chart, decision boundary
```

Convention throughout: `Z = X . W + b`, where `X` is (batch, inputs), `W` is (inputs, outputs)
and `b` is a (1, outputs) row broadcast down the batch.
