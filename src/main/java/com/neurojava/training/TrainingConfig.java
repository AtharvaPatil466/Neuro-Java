package com.neurojava.training;

/**
 * Hyperparameters for one training run.
 *
 * @param epochs      full passes over the training set
 * @param learningRate step size for gradient descent
 * @param batchSize   rows per gradient update; use the training-set size for full-batch
 * @param seed        makes weight init and shuffling reproducible
 */
public record TrainingConfig(int epochs, double learningRate, int batchSize, long seed) {

    /** Highest step size that still stands a chance of converging; above it the run just explodes. */
    public static final double MAX_LEARNING_RATE = 10.0;

    /** Rejects non-positive epochs/batch size and a learning rate outside (0, 10]. */
    public TrainingConfig {
        if (epochs <= 0) {
            throw new IllegalArgumentException("epochs must be positive but was " + epochs);
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive but was " + batchSize);
        }
        if (!(learningRate > 0.0) || learningRate > MAX_LEARNING_RATE) {
            throw new IllegalArgumentException(
                    "learningRate must be in (0, " + MAX_LEARNING_RATE + "] but was " + learningRate);
        }
    }
}
