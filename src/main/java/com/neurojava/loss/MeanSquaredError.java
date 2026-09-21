package com.neurojava.loss;

import com.neurojava.core.Matrix;

/** Mean squared error — the regression default. */
public final class MeanSquaredError implements LossFunction {

    /** The 1/2 in the formula is what makes the derivative come out as a clean (p - t). */
    private static final double HALVING_FACTOR = 2.0;

    @Override
    public String name() {
        return "Mean Squared Error";
    }

    @Override
    public double value(Matrix predicted, Matrix target) {
        requireSameShape(predicted, target);
        Matrix diff = predicted.subtract(target);
        return diff.hadamard(diff).sum() / (HALVING_FACTOR * predicted.rows());
    }

    @Override
    public Matrix gradient(Matrix predicted, Matrix target) {
        requireSameShape(predicted, target);
        return predicted.subtract(target).scale(1.0 / predicted.rows());
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
