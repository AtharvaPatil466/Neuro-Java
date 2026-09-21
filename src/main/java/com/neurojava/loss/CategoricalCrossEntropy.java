package com.neurojava.loss;

import com.neurojava.core.Matrix;

/** Categorical cross entropy, for softmax outputs with one-hot targets. */
public final class CategoricalCrossEntropy implements LossFunction {

    /** Keeps ln(p) finite when the network drives a probability to exactly 0. */
    private static final double EPSILON = 1e-12;

    @Override
    public String name() {
        return "Categorical Cross Entropy";
    }

    @Override
    public double value(Matrix predicted, Matrix target) {
        requireSameShape(predicted, target);
        double total = 0.0;
        for (int r = 0; r < predicted.rows(); r++) {
            for (int c = 0; c < predicted.cols(); c++) {
                total += target.get(r, c) * Math.log(Math.max(EPSILON, predicted.get(r, c)));
            }
        }
        return -total / predicted.rows();
    }

    /** dL/dp = -t / p / N — the exact derivative of {@link #value}, same clamp. */
    @Override
    public Matrix gradient(Matrix predicted, Matrix target) {
        requireSameShape(predicted, target);
        int n = predicted.rows();
        double[][] out = new double[n][predicted.cols()];
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < predicted.cols(); c++) {
                out[r][c] = -target.get(r, c) / (Math.max(EPSILON, predicted.get(r, c)) * n);
            }
        }
        return new Matrix(out);
    }

    private static void requireSameShape(Matrix predicted, Matrix target) {
        if (predicted.rows() != target.rows() || predicted.cols() != target.cols()) {
            throw new IllegalArgumentException(
                    "predicted " + predicted.shape() + " and target " + target.shape()
                            + " must have the same shape");
        }
        if (predicted.rows() == 0) {
            throw new IllegalArgumentException("cannot average a loss over an empty batch " + predicted.shape());
        }
    }
}
