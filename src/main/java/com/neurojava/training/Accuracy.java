package com.neurojava.training;

import com.neurojava.core.Matrix;

import java.util.Objects;

/** Correct-prediction rate. */
public final class Accuracy {

    /** Anything at or above this counts as class 1 in a single-output network. */
    private static final double BINARY_THRESHOLD = 0.5;

    private Accuracy() {}

    /**
     * Single-column targets are thresholded at 0.5; multi-column targets compare argmax.
     * Returns NaN for an empty batch.
     */
    public static double of(Matrix predicted, Matrix target) {
        Objects.requireNonNull(predicted, "predicted must not be null");
        Objects.requireNonNull(target, "target must not be null");
        if (predicted.rows() != target.rows() || predicted.cols() != target.cols()) {
            throw new IllegalArgumentException(
                    "predicted " + predicted.shape() + " and target " + target.shape() + " must have the same shape");
        }
        int rows = predicted.rows();
        if (rows == 0) {
            return Double.NaN;
        }
        int correct = 0;
        for (int r = 0; r < rows; r++) {
            if (isCorrect(predicted, target, r)) {
                correct++;
            }
        }
        return (double) correct / rows;
    }

    private static boolean isCorrect(Matrix predicted, Matrix target, int row) {
        if (predicted.cols() == 1) {
            boolean predictedClass = predicted.get(row, 0) >= BINARY_THRESHOLD;
            boolean targetClass = target.get(row, 0) >= BINARY_THRESHOLD;
            return predictedClass == targetClass;
        }
        return predicted.argMaxInRow(row) == target.argMaxInRow(row);
    }
}
