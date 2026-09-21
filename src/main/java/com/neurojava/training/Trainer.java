package com.neurojava.training;

import com.neurojava.core.Matrix;
import com.neurojava.nn.NeuralNetwork;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

/**
 * Runs the epoch loop: forward, loss, backward, update, report.
 *
 * <p>Plain Java with no JavaFX dependency — callers marshal {@link Listener} callbacks onto
 * their own UI thread. {@link #train} blocks, so the UI runs it on a background thread and
 * steers it with {@link #pause}, {@link #resume} and {@link #stop}.
 */
public final class Trainer {

    /** How often a paused run wakes up to re-check the flags. */
    private static final long PAUSE_POLL_MILLIS = 50L;

    /** Receives one callback per epoch, on the thread that called {@link #train}. */
    public interface Listener {
        void onEpoch(Metrics metrics);
    }

    private final NeuralNetwork network;
    private final TrainingConfig config;

    private volatile boolean running;
    private volatile boolean paused;
    private volatile boolean stopRequested;
    private volatile boolean diverged;

    public Trainer(NeuralNetwork network, TrainingConfig config) {
        this.network = Objects.requireNonNull(network, "network must not be null");
        this.config = Objects.requireNonNull(config, "config must not be null");
    }

    /**
     * Trains for {@code config.epochs()}, or until {@link #stop} is called or the loss
     * becomes non-finite (diverged).
     *
     * @param valInputs may be a zero-row matrix, in which case validation accuracy is NaN
     * @param listener  may be null
     */
    public void train(Matrix trainInputs,
                      Matrix trainTargets,
                      Matrix valInputs,
                      Matrix valTargets,
                      Listener listener) {
        requireSameRows(trainInputs, trainTargets, "training");
        requireSameRows(valInputs, valTargets, "validation");
        if (trainInputs.rows() == 0) {
            throw new IllegalArgumentException("training set is empty");
        }

        running = true;
        paused = false;
        stopRequested = false;
        diverged = false;
        long startedAt = System.currentTimeMillis();
        int batchSize = Math.min(config.batchSize(), trainInputs.rows());

        try {
            for (int epoch = 0; epoch < config.epochs(); epoch++) {
                if (!awaitResume()) {
                    return;
                }
                double meanLoss = runEpoch(trainInputs, trainTargets, epoch, batchSize);
                report(listener, epoch, meanLoss, trainInputs, trainTargets, valInputs, valTargets, startedAt);
                if (diverged || stopRequested) {
                    return;
                }
            }
        } finally {
            running = false;
        }
    }

    /** One shuffled pass over the training set; returns the sample-weighted mean batch loss. */
    private double runEpoch(Matrix inputs, Matrix targets, int epoch, int batchSize) {
        int sampleCount = inputs.rows();
        int[] order = shuffledIndices(sampleCount, new Random(config.seed() + epoch));
        double weightedLossSum = 0.0;
        int processed = 0;

        for (int from = 0; from < sampleCount; from += batchSize) {
            if (!awaitResume()) {
                break;
            }
            int[] batch = Arrays.copyOfRange(order, from, Math.min(from + batchSize, sampleCount));
            Matrix predicted = network.forward(inputs.selectRows(batch));
            if (!predicted.isFinite()) {
                diverged = true;
                return Double.NaN;
            }
            double batchLoss = network.backward(predicted, targets.selectRows(batch));
            if (!Double.isFinite(batchLoss)) {
                diverged = true;
                return Double.NaN;
            }
            network.updateWeights(config.learningRate());
            weightedLossSum += batchLoss * batch.length;
            processed += batch.length;
        }
        return processed == 0 ? Double.NaN : weightedLossSum / processed;
    }

    private void report(Listener listener,
                        int epoch,
                        double meanLoss,
                        Matrix trainInputs,
                        Matrix trainTargets,
                        Matrix valInputs,
                        Matrix valTargets,
                        long startedAt) {
        if (listener == null) {
            return;
        }
        double trainAccuracy = Accuracy.of(network.predict(trainInputs), trainTargets);
        double valAccuracy = valInputs.rows() > 0
                ? Accuracy.of(network.predict(valInputs), valTargets)
                : Double.NaN;
        listener.onEpoch(new Metrics(epoch + 1,
                config.epochs(),
                meanLoss,
                trainAccuracy,
                valAccuracy,
                System.currentTimeMillis() - startedAt));
    }

    /** Blocks while paused. Returns false when the run should end (stopped or interrupted). */
    private boolean awaitResume() {
        while (paused && !stopRequested) {
            try {
                Thread.sleep(PAUSE_POLL_MILLIS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return !stopRequested;
    }

    /** Fisher-Yates over the row indices, so a batch is a random sample without replacement. */
    private static int[] shuffledIndices(int count, Random rng) {
        int[] indices = new int[count];
        for (int i = 0; i < count; i++) {
            indices[i] = i;
        }
        for (int i = count - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int swap = indices[i];
            indices[i] = indices[j];
            indices[j] = swap;
        }
        return indices;
    }

    private static void requireSameRows(Matrix inputs, Matrix targets, String what) {
        Objects.requireNonNull(inputs, what + " inputs must not be null");
        Objects.requireNonNull(targets, what + " targets must not be null");
        if (inputs.rows() != targets.rows()) {
            throw new IllegalArgumentException(what + " inputs " + inputs.shape()
                    + " and targets " + targets.shape() + " have different row counts");
        }
    }

    public void pause() {
        paused = true;
    }

    public void resume() {
        paused = false;
    }

    /** Asks the loop to finish after the current epoch. */
    public void stop() {
        stopRequested = true;
        paused = false;
    }

    public boolean isPaused() {
        return paused;
    }

    public boolean isRunning() {
        return running;
    }

    /** Set when training halted because the loss went NaN or infinite. */
    public boolean hasDiverged() {
        return diverged;
    }
}
