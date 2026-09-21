package com.neurojava.experiments;

import com.neurojava.activation.ActivationFunction;
import com.neurojava.activation.ReLU;
import com.neurojava.activation.Sigmoid;
import com.neurojava.activation.Softmax;
import com.neurojava.core.Matrix;
import com.neurojava.data.Dataset;
import com.neurojava.data.IrisDataset;
import com.neurojava.data.Split;
import com.neurojava.data.XorDataset;
import com.neurojava.loss.BinaryCrossEntropy;
import com.neurojava.loss.CategoricalCrossEntropy;
import com.neurojava.loss.LossFunction;
import com.neurojava.nn.DenseLayer;
import com.neurojava.nn.NeuralNetwork;
import com.neurojava.training.Accuracy;
import com.neurojava.training.Metrics;
import com.neurojava.training.Trainer;
import com.neurojava.training.TrainingConfig;

import java.util.Random;
import java.util.function.Supplier;

/**
 * Headless sweeps for the project report: how learning rate, hidden width, activation and
 * epoch budget actually affect learning.
 *
 * <p>Every number printed is measured here and now — nothing is hard-coded. Each configuration
 * is repeated over {@link #REPEATS} seeds and averaged, because a single run of a randomly
 * initialised network says more about the seed than about the hyperparameter.
 *
 * <p>Run with: {@code java -cp target/classes com.neurojava.experiments.ExperimentRunner}
 */
public final class ExperimentRunner {

    private static final int REPEATS = 5;
    private static final double TRAIN_FRACTION = 0.8;
    private static final long BASE_SEED = 42L;

    private ExperimentRunner() {
    }

    /** Averaged outcome of one configuration. */
    private record Result(double finalLoss, double trainAccuracy, double valAccuracy, double millis) {
    }

    public static void main(String[] args) {
        System.out.println("NeuroJava — experimental results");
        System.out.println("Each row is the mean of " + REPEATS + " runs with different seeds.\n");

        learningRateSweep();
        hiddenWidthSweep();
        activationComparison();
        epochBudgetSweep();
        irisBaseline();

        System.out.println("\nDone. These tables are the measured results for section 37 of the report.");
    }

    private static void learningRateSweep() {
        header("Experiment 1 — learning rate (XOR, 2 -> 4 -> 1, sigmoid, 1000 epochs)",
                "Learning rate");
        for (double lr : new double[]{0.001, 0.01, 0.1, 0.5}) {
            Result r = average(XorDataset::new, new int[]{4}, Sigmoid::new, lr, 1000, 4);
            row(String.format("%.3f", lr), r);
        }
        System.out.println();
    }

    private static void hiddenWidthSweep() {
        header("Experiment 2 — hidden neurons (XOR, sigmoid, lr 0.1, 1000 epochs)", "Architecture");
        for (int hidden : new int[]{2, 4, 8, 16}) {
            Result r = average(XorDataset::new, new int[]{hidden}, Sigmoid::new, 0.1, 1000, 4);
            row("2 -> " + hidden + " -> 1", r);
        }
        System.out.println();
    }

    private static void activationComparison() {
        header("Experiment 3 — activation function (XOR, 2 -> 4 -> 1, lr 0.1, 1000 epochs)",
                "Activation");
        row("Sigmoid", average(XorDataset::new, new int[]{4}, Sigmoid::new, 0.1, 1000, 4));
        row("ReLU", average(XorDataset::new, new int[]{4}, ReLU::new, 0.1, 1000, 4));
        System.out.println();
    }

    private static void epochBudgetSweep() {
        header("Experiment 4 — epoch budget (XOR, 2 -> 4 -> 1, sigmoid, lr 0.1)", "Epochs");
        for (int epochs : new int[]{100, 500, 1000, 2000}) {
            Result r = average(XorDataset::new, new int[]{4}, Sigmoid::new, 0.1, epochs, 4);
            row(Integer.toString(epochs), r);
        }
        System.out.println();
    }

    private static void irisBaseline() {
        header("Experiment 5 — Iris multiclass (4 -> h -> 3, ReLU hidden + softmax head, lr 0.05, 500 epochs)",
                "Architecture");
        for (int[] hidden : new int[][]{{6}, {8}, {8, 6}}) {
            Result r = average(IrisDataset::new, hidden, ReLU::new, 0.05, 500, 16);
            StringBuilder label = new StringBuilder("4");
            for (int h : hidden) {
                label.append(" -> ").append(h);
            }
            row(label.append(" -> 3").toString(), r);
        }
        System.out.println();
    }

    /** Runs one configuration over {@link #REPEATS} seeds and averages the outcome. */
    private static Result average(Supplier<Dataset> datasetFactory,
                                  int[] hiddenSizes,
                                  Supplier<ActivationFunction> hiddenActivation,
                                  double learningRate,
                                  int epochs,
                                  int batchSize) {
        double loss = 0.0;
        double trainAccuracy = 0.0;
        double valAccuracy = 0.0;
        double millis = 0.0;

        for (int repeat = 0; repeat < REPEATS; repeat++) {
            long seed = BASE_SEED + repeat;
            Dataset dataset = datasetFactory.get();
            // XOR has only four samples, so holding any of them back would be meaningless.
            double trainFraction = dataset.inputs().rows() < 10 ? 1.0 : TRAIN_FRACTION;
            Split split = Split.of(dataset, trainFraction, new Random(seed));

            NeuralNetwork network = build(dataset, hiddenSizes, hiddenActivation, seed);
            Trainer trainer = new Trainer(network, new TrainingConfig(epochs, learningRate, batchSize, seed));

            Metrics[] last = new Metrics[1];
            trainer.train(split.trainInputs(), split.trainTargets(),
                    split.valInputs(), split.valTargets(), m -> last[0] = m);

            Metrics metrics = last[0];
            loss += metrics.loss();
            trainAccuracy += metrics.trainAccuracy();
            valAccuracy += Double.isNaN(metrics.valAccuracy())
                    ? Accuracy.of(network.predict(split.trainInputs()), split.trainTargets())
                    : metrics.valAccuracy();
            millis += metrics.elapsedMillis();
        }
        return new Result(loss / REPEATS, trainAccuracy / REPEATS, valAccuracy / REPEATS, millis / REPEATS);
    }

    /**
     * Same head rule the UI uses: a binary dataset gets one sigmoid output with binary cross
     * entropy, anything else gets a softmax over one-hot targets with categorical cross entropy.
     */
    private static NeuralNetwork build(Dataset dataset,
                                       int[] hiddenSizes,
                                       Supplier<ActivationFunction> hiddenActivation,
                                       long seed) {
        Random rng = new Random(seed);
        LossFunction loss = dataset.isBinary() ? new BinaryCrossEntropy() : new CategoricalCrossEntropy();
        NeuralNetwork network = new NeuralNetwork(loss);

        int previous = dataset.inputSize();
        for (int size : hiddenSizes) {
            network.add(new DenseLayer(previous, size, hiddenActivation.get(), rng));
            previous = size;
        }
        ActivationFunction head = dataset.isBinary() ? new Sigmoid() : new Softmax();
        network.add(new DenseLayer(previous, dataset.outputSize(), head, rng));
        return network;
    }

    private static void header(String title, String firstColumn) {
        System.out.println("## " + title);
        System.out.printf("| %-28s | Final loss | Train acc | Val acc | Time (ms) |%n", firstColumn);
        System.out.println("|------------------------------|------------|-----------|---------|-----------|");
    }

    private static void row(String label, Result r) {
        System.out.printf("| %-28s | %10.5f | %8.1f%% | %6.1f%% | %9.0f |%n",
                label, r.finalLoss(), r.trainAccuracy() * 100, r.valAccuracy() * 100, r.millis());
    }
}
