package com.neurojava.loss;

import com.neurojava.core.Matrix;

/** Binary cross entropy, for sigmoid outputs with 0/1 targets. */
public final class BinaryCrossEntropy implements LossFunction {

    /** Keeps ln(p) and ln(1 - p) finite when the network saturates at exactly 0 or 1. */
    private static final double EPSILON = 1e-12;

    @Override
    public String name() {
        return "Binary Cross Entropy";
    }

    @Override
    public double value(Matrix predicted, Matrix target) {
        requireSameShape(predicted, target);
        double total = 0.0;
        for (int r = 0; r < predicted.rows(); r++) {
            for (int c = 0; c < predicted.cols(); c++) {
                double p = clamp(predicted.get(r, c));
                double t = target.get(r, c);
                total += t * Math.log(p) + (1.0 - t) * Math.log(1.0 - p);
            }
        }
        return -total / predicted.rows();
    }

    /** dL/dp = (p - t) / (p * (1 - p)) / N — the exact derivative of {@link #value}, same clamp. */
    @Override
    public Matrix gradient(Matrix predicted, Matrix target) {
        requireSameShape(predicted, target);
        int n = predicted.rows();
        double[][] out = new double[n][predicted.cols()];
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < predicted.cols(); c++) {
                double p = clamp(predicted.get(r, c));
                double t = target.get(r, c);
                out[r][c] = (p - t) / (p * (1.0 - p) * n);
            }
        }
        return new Matrix(out);
    }

    private static double clamp(double p) {
        return Math.min(1.0 - EPSILON, Math.max(EPSILON, p));
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
