package com.neurojava.activation;

import com.neurojava.core.Matrix;

/** Row-wise softmax, turning each row of scores into a probability distribution. */
public final class Softmax implements ActivationFunction {

    @Override
    public String name() {
        return "Softmax";
    }

    @Override
    public String formula() {
        return "f(x_i) = e^(x_i) / sum_j e^(x_j)";
    }

    /**
     * Subtracting the row maximum before exponentiating leaves the result unchanged
     * (the constant cancels top and bottom) but keeps every exponent &lt;= 0, so nothing overflows.
     */
    @Override
    public Matrix apply(Matrix z) {
        double[][] out = z.toArray();
        for (double[] row : out) {
            if (row.length == 0) {
                continue;
            }
            double max = Double.NEGATIVE_INFINITY;
            for (double v : row) {
                max = Math.max(max, v);
            }
            double sum = 0.0;
            for (int c = 0; c < row.length; c++) {
                row[c] = Math.exp(row[c] - max);
                sum += row[c];
            }
            for (int c = 0; c < row.length; c++) {
                row[c] /= sum;
            }
        }
        return new Matrix(out);
    }

    /**
     * True Jacobian-vector product, one row at a time:
     * dZ_i = a_i * (gradA_i - sum_j gradA_j * a_j).
     *
     * <p>Deliberately NOT fused with cross-entropy into the famous (a - y) shortcut. Doing the
     * chain rule honestly is the academic point of this project: softmax stays a self-contained
     * activation that works with any loss, and MathCheck can verify the composed
     * softmax -> cross-entropy gradient against finite differences.
     */
    @Override
    public Matrix gradient(Matrix z, Matrix a, Matrix gradA) {
        if (a.rows() != gradA.rows() || a.cols() != gradA.cols()) {
            throw new IllegalArgumentException(
                    "softmax output " + a.shape() + " and incoming gradient " + gradA.shape()
                            + " must have the same shape");
        }
        double[][] out = new double[a.rows()][a.cols()];
        for (int r = 0; r < a.rows(); r++) {
            double dot = 0.0;
            for (int c = 0; c < a.cols(); c++) {
                dot += gradA.get(r, c) * a.get(r, c);
            }
            for (int c = 0; c < a.cols(); c++) {
                out[r][c] = a.get(r, c) * (gradA.get(r, c) - dot);
            }
        }
        return new Matrix(out);
    }
}
