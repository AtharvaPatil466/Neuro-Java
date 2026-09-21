package com.neurojava.nn;

import com.neurojava.activation.ActivationFunction;
import com.neurojava.core.Matrix;
import com.neurojava.loss.LossFunction;
import com.neurojava.training.Accuracy;
import com.neurojava.training.Metrics;
import com.neurojava.training.Trainer;
import com.neurojava.training.TrainingConfig;

import java.util.Random;

/**
 * Self-check for the layer / network / trainer plumbing — run it with
 * {@code java com.neurojava.nn.GradientCheck}.
 *
 * <p>It deliberately uses its own inline sigmoid and squared-error loss so that it tests only
 * this package's maths, independent of the real activation and loss implementations.
 */
public final class GradientCheck {

    private static final double STEP = 1e-6;
    private static final double RELATIVE_TOLERANCE = 1e-5;

    private GradientCheck() {}

    public static void main(String[] args) {
        checkAnalyticVsNumericalGradients();
        checkLearningHappens();
        checkGuardRails();
        checkReproducibility();
        System.out.println("PASS all 4 checks");
    }

    // ---------------------------------------------------------------- check 1

    /** Central finite differences on every weight and bias vs what backward() produced. */
    private static void checkAnalyticVsNumericalGradients() {
        Random rng = new Random(7L);
        LossFunction loss = new SquaredError();
        NeuralNetwork network = new NeuralNetwork(loss)
                .add(new DenseLayer(3, 4, new InlineSigmoid(), rng))
                .add(new DenseLayer(4, 2, new InlineSigmoid(), rng));

        Matrix inputs = new Matrix(new double[][]{
                {0.5, -1.2, 0.3},
                {-0.7, 0.9, 1.4},
                {1.1, 0.2, -0.6},
                {-0.3, -0.8, 0.7},
        });
        Matrix targets = new Matrix(new double[][]{
                {1.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {0.0, 1.0},
        });

        network.backward(network.forward(inputs), targets);

        int compared = 0;
        double worst = 0.0;
        for (int index = 0; index < network.layers().size(); index++) {
            DenseLayer layer = (DenseLayer) network.layers().get(index);
            Matrix analyticWeights = layer.gradWeights();
            Matrix analyticBiases = layer.gradBiases();
            Matrix weights = layer.weights();
            Matrix biases = layer.biases();

            for (int r = 0; r < weights.rows(); r++) {
                for (int c = 0; c < weights.cols(); c++) {
                    double numeric = numericWeightGradient(network, layer, weights, biases, r, c, inputs, targets);
                    worst = Math.max(worst, compare(analyticWeights.get(r, c), numeric,
                            "layer " + index + " weight[" + r + "][" + c + "]"));
                    compared++;
                }
            }
            for (int c = 0; c < biases.cols(); c++) {
                double numeric = numericBiasGradient(network, layer, weights, biases, c, inputs, targets);
                worst = Math.max(worst, compare(analyticBiases.get(0, c), numeric,
                        "layer " + index + " bias[" + c + "]"));
                compared++;
            }
            layer.setParameters(weights, biases);
        }
        System.out.printf("PASS backpropagation: %d analytic gradients match central differences "
                + "(worst relative error %.2e, tolerance %.0e)%n", compared, worst, RELATIVE_TOLERANCE);
    }

    private static double numericWeightGradient(NeuralNetwork network,
                                                DenseLayer layer,
                                                Matrix weights,
                                                Matrix biases,
                                                int row,
                                                int col,
                                                Matrix inputs,
                                                Matrix targets) {
        layer.setParameters(nudge(weights, row, col, STEP), biases);
        double up = lossOf(network, inputs, targets);
        layer.setParameters(nudge(weights, row, col, -STEP), biases);
        double down = lossOf(network, inputs, targets);
        layer.setParameters(weights, biases);
        return (up - down) / (2.0 * STEP);
    }

    private static double numericBiasGradient(NeuralNetwork network,
                                              DenseLayer layer,
                                              Matrix weights,
                                              Matrix biases,
                                              int col,
                                              Matrix inputs,
                                              Matrix targets) {
        layer.setParameters(weights, nudge(biases, 0, col, STEP));
        double up = lossOf(network, inputs, targets);
        layer.setParameters(weights, nudge(biases, 0, col, -STEP));
        double down = lossOf(network, inputs, targets);
        layer.setParameters(weights, biases);
        return (up - down) / (2.0 * STEP);
    }

    private static double lossOf(NeuralNetwork network, Matrix inputs, Matrix targets) {
        return network.loss().value(network.forward(inputs), targets);
    }

    /** Copy of {@code m} with one element shifted by {@code delta} — Matrix itself stays immutable. */
    private static Matrix nudge(Matrix m, int row, int col, double delta) {
        double[][] values = m.toArray();
        values[row][col] += delta;
        return new Matrix(values);
    }

    private static double compare(double analytic, double numeric, String what) {
        double relativeError = Math.abs(analytic - numeric)
                / Math.max(1.0, Math.max(Math.abs(analytic), Math.abs(numeric)));
        check(relativeError <= RELATIVE_TOLERANCE,
                what + ": analytic " + analytic + " vs numeric " + numeric
                        + " (relative error " + relativeError + ")");
        return relativeError;
    }

    // ---------------------------------------------------------------- check 2

    private static void checkLearningHappens() {
        XorRun run = trainXor(42L);
        check(run.finalLoss < run.initialLoss * 0.1,
                "XOR loss barely moved: " + run.initialLoss + " -> " + run.finalLoss);
        check(run.accuracy == 1.0, "XOR accuracy reached only " + run.accuracy);
        check(run.epochs > 0, "no epoch was reported");
        System.out.printf("PASS learning: XOR loss %.4f -> %.6f over %d epochs, accuracy %.0f%%%n",
                run.initialLoss, run.finalLoss, run.epochs, run.accuracy * 100.0);
    }

    // ---------------------------------------------------------------- check 3

    private static void checkGuardRails() {
        NeuralNetwork network = new NeuralNetwork(new SquaredError())
                .add(new DenseLayer(2, 4, new InlineSigmoid(), new Random(1L)));
        expectRejected("NeuralNetwork.add with a 3-input layer after a 4-output layer",
                () -> network.add(new DenseLayer(3, 1, new InlineSigmoid(), new Random(1L))));
        expectRejected("forward with the wrong feature count",
                () -> network.forward(Matrix.rowVector(1.0, 2.0, 3.0)));
        expectRejected("epochs = 0", () -> new TrainingConfig(0, 0.1, 4, 1L));
        expectRejected("negative learning rate", () -> new TrainingConfig(10, -0.5, 4, 1L));
        expectRejected("batchSize = 0", () -> new TrainingConfig(10, 0.1, 0, 1L));

        check(Double.isNaN(Accuracy.of(Matrix.zeros(0, 1), Matrix.zeros(0, 1))),
                "empty batch accuracy should be NaN");
        double binary = Accuracy.of(new Matrix(new double[][]{{0.9}, {0.2}, {0.6}, {0.4}}),
                new Matrix(new double[][]{{1.0}, {0.0}, {0.0}, {0.0}}));
        check(binary == 0.75, "single-column accuracy should be 0.75 but was " + binary);
        double argmax = Accuracy.of(new Matrix(new double[][]{{0.1, 0.7, 0.2}, {0.8, 0.1, 0.1}}),
                new Matrix(new double[][]{{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}));
        check(argmax == 0.5, "argmax accuracy should be 0.5 but was " + argmax);
        expectRejected("accuracy on mismatched shapes",
                () -> Accuracy.of(Matrix.zeros(2, 3), Matrix.zeros(2, 2)));

        int stoppedAt = runUntilStopped(4);
        check(stoppedAt == 4, "stop() should have ended the run at epoch 4 but it reached " + stoppedAt);
        int divergedAt = runUntilDiverged();
        System.out.println("PASS guard rails: bad shapes, bad configs, stop() and divergence all handled ("
                + "long run halted at epoch " + stoppedAt + " of 100000, NaN run caught at epoch "
                + divergedAt + ")");
    }

    /** A linear net on large inputs at the maximum learning rate blows up within a few epochs. */
    private static int runUntilDiverged() {
        Random rng = new Random(3L);
        NeuralNetwork network = new NeuralNetwork(new SquaredError())
                .add(new DenseLayer(2, 4, new InlineLinear(), rng))
                .add(new DenseLayer(4, 1, new InlineLinear(), rng));
        int epochs = 500;
        Trainer trainer = new Trainer(network, new TrainingConfig(epochs, TrainingConfig.MAX_LEARNING_RATE, 4, 5L));
        Metrics[] last = {null};
        trainer.train(xorInputs().scale(50.0), xorTargets().scale(50.0),
                Matrix.zeros(0, 2), Matrix.zeros(0, 1), m -> last[0] = m);
        check(trainer.hasDiverged(), "a run whose loss went non-finite was not flagged as diverged");
        check(last[0] != null, "no final Metrics was emitted for the diverged run");
        check(last[0].epoch() < epochs,
                "diverged run should have stopped early but ran all " + epochs + " epochs");
        check(!trainer.isRunning(), "trainer still reports running after diverging");
        return last[0].epoch();
    }

    /** Runs a deliberately endless training job and stops it from inside the listener. */
    private static int runUntilStopped(int stopAfterEpochs) {
        NeuralNetwork network = buildXorNetwork(1L);
        Trainer trainer = new Trainer(network, new TrainingConfig(100_000, 0.5, 4, 1L));
        int[] lastEpoch = {0};
        trainer.train(xorInputs(), xorTargets(), Matrix.zeros(0, 2), Matrix.zeros(0, 1), metrics -> {
            lastEpoch[0] = metrics.epoch();
            if (metrics.epoch() >= stopAfterEpochs) {
                trainer.stop();
            }
        });
        check(!trainer.isRunning(), "trainer still reports running after train() returned");
        check(!trainer.hasDiverged(), "stopped run should not be marked diverged");
        return lastEpoch[0];
    }

    // ---------------------------------------------------------------- check 4

    private static void checkReproducibility() {
        XorRun first = trainXor(2026L);
        XorRun second = trainXor(2026L);
        check(first.finalLoss == second.finalLoss,
                "same seed gave different losses: " + first.finalLoss + " vs " + second.finalLoss);
        System.out.printf("PASS reproducibility: seed 2026 gives loss %.10f twice%n", first.finalLoss);
    }

    // ---------------------------------------------------------------- XOR run

    private static final class XorRun {
        final double initialLoss;
        final double finalLoss;
        final double accuracy;
        final int epochs;

        XorRun(double initialLoss, double finalLoss, double accuracy, int epochs) {
            this.initialLoss = initialLoss;
            this.finalLoss = finalLoss;
            this.accuracy = accuracy;
            this.epochs = epochs;
        }
    }

    private static XorRun trainXor(long seed) {
        NeuralNetwork network = buildXorNetwork(seed);
        Matrix inputs = xorInputs();
        Matrix targets = xorTargets();
        double initialLoss = lossOf(network, inputs, targets);

        Trainer trainer = new Trainer(network, new TrainingConfig(4000, 1.0, 4, seed));
        Metrics[] last = {null};
        trainer.train(inputs, targets, Matrix.zeros(0, 2), Matrix.zeros(0, 1), m -> last[0] = m);
        check(last[0] != null, "trainer emitted no metrics");
        check(!trainer.hasDiverged(), "XOR run diverged");

        double accuracy = Accuracy.of(network.predict(inputs), targets);
        return new XorRun(initialLoss, last[0].loss(), accuracy, last[0].epoch());
    }

    private static NeuralNetwork buildXorNetwork(long seed) {
        Random rng = new Random(seed);
        return new NeuralNetwork(new SquaredError())
                .add(new DenseLayer(2, 4, new InlineSigmoid(), rng))
                .add(new DenseLayer(4, 1, new InlineSigmoid(), rng));
    }

    private static Matrix xorInputs() {
        return new Matrix(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    }

    private static Matrix xorTargets() {
        return new Matrix(new double[][]{{0}, {1}, {1}, {0}});
    }

    // ---------------------------------------------------------------- helpers

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException("CHECK FAILED - " + message);
        }
    }

    private static void expectRejected(String what, Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException | IllegalStateException expected) {
            return;
        }
        throw new IllegalStateException("CHECK FAILED - " + what + " was accepted but must be rejected");
    }

    /** Local sigmoid so this check never depends on the activation package's implementations. */
    private static final class InlineSigmoid implements ActivationFunction {
        @Override public String name() { return "InlineSigmoid"; }

        @Override public String formula() { return "f(x) = 1 / (1 + e^-x)"; }

        @Override public Matrix apply(Matrix z) { return z.map(v -> 1.0 / (1.0 + Math.exp(-v))); }

        @Override public Matrix gradient(Matrix z, Matrix a, Matrix gradA) {
            return gradA.hadamard(a.map(v -> v * (1.0 - v)));
        }
    }

    /** Local identity activation — used only to make a run diverge on purpose. */
    private static final class InlineLinear implements ActivationFunction {
        @Override public String name() { return "InlineLinear"; }

        @Override public String formula() { return "f(x) = x"; }

        @Override public Matrix apply(Matrix z) { return z; }

        @Override public Matrix gradient(Matrix z, Matrix a, Matrix gradA) { return gradA; }
    }

    /** Local squared error: mean over samples of the summed squared residuals. */
    private static final class SquaredError implements LossFunction {
        @Override public String name() { return "Inline Squared Error"; }

        @Override public double value(Matrix predicted, Matrix target) {
            Matrix residual = predicted.subtract(target);
            return residual.hadamard(residual).sum() / predicted.rows();
        }

        @Override public Matrix gradient(Matrix predicted, Matrix target) {
            return predicted.subtract(target).scale(2.0 / predicted.rows());
        }
    }
}
