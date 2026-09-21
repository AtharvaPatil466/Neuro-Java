package com.neurojava.core;

import java.util.Objects;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * Immutable row-major matrix. Every operation returns a NEW Matrix; nothing is mutated in place.
 *
 * <p>Convention used throughout NeuroJava: one ROW per sample, one COLUMN per feature.
 * A dense layer therefore computes {@code Z = X . W + b} where X is (batch, in),
 * W is (in, out) and b is a (1, out) row that broadcasts down the batch.
 */
public final class Matrix {

    private final double[][] data;
    private final int cols;

    /** Wraps a defensive copy of {@code data}. Rows must all be the same length. */
    public Matrix(double[][] data) {
        Objects.requireNonNull(data, "data must not be null");
        int width = data.length == 0 ? 0 : data[0].length;
        double[][] copy = new double[data.length][];
        for (int r = 0; r < data.length; r++) {
            if (data[r] == null) {
                throw new IllegalArgumentException("row " + r + " is null");
            }
            if (data[r].length != width) {
                throw new IllegalArgumentException(
                        "ragged matrix: row 0 has " + width + " columns but row " + r + " has " + data[r].length);
            }
            copy[r] = data[r].clone();
        }
        this.data = copy;
        this.cols = width;
    }

    /** Takes ownership of {@code data} without copying. Callers must never touch it again. */
    private Matrix(double[][] data, int cols) {
        this.data = data;
        this.cols = cols;
    }

    public static Matrix zeros(int rows, int cols) {
        requireNonNegative(rows, "rows");
        requireNonNegative(cols, "cols");
        return new Matrix(new double[rows][cols], cols);
    }

    /** Uniform random values in {@code [-scale, scale]}. */
    public static Matrix random(int rows, int cols, double scale, Random rng) {
        Objects.requireNonNull(rng, "rng must not be null");
        requireNonNegative(rows, "rows");
        requireNonNegative(cols, "cols");
        double[][] out = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out[r][c] = (rng.nextDouble() * 2.0 - 1.0) * scale;
            }
        }
        return new Matrix(out, cols);
    }

    /** Single-row matrix, useful for one sample. */
    public static Matrix rowVector(double... values) {
        Objects.requireNonNull(values, "values must not be null");
        return new Matrix(new double[][]{values.clone()}, values.length);
    }

    public int rows() {
        return data.length;
    }

    public int cols() {
        return cols;
    }

    public double get(int row, int col) {
        return data[row][col];
    }

    /** Defensive copy of the backing array. */
    public double[][] toArray() {
        double[][] copy = new double[data.length][];
        for (int r = 0; r < data.length; r++) {
            copy[r] = data[r].clone();
        }
        return copy;
    }

    /** Row {@code i} as a (1, cols) matrix. */
    public Matrix row(int i) {
        return new Matrix(new double[][]{data[i].clone()}, cols);
    }

    /** Sub-matrix of rows {@code [from, to)} — used for mini-batches. */
    public Matrix rowRange(int from, int to) {
        if (from < 0 || to > data.length || from > to) {
            throw new IndexOutOfBoundsException(
                    "row range [" + from + ", " + to + ") outside 0.." + data.length);
        }
        double[][] out = new double[to - from][];
        for (int r = from; r < to; r++) {
            out[r - from] = data[r].clone();
        }
        return new Matrix(out, cols);
    }

    /** Rows picked by index, in the given order — used for shuffled mini-batches. */
    public Matrix selectRows(int[] indices) {
        Objects.requireNonNull(indices, "indices must not be null");
        double[][] out = new double[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            out[i] = data[indices[i]].clone();
        }
        return new Matrix(out, cols);
    }

    /** Matrix product: (n, m) . (m, p) -> (n, p). */
    public Matrix multiply(Matrix other) {
        Objects.requireNonNull(other, "other must not be null");
        if (cols != other.data.length) {
            throw new IllegalArgumentException(
                    "cannot multiply " + shape() + " by " + other.shape() + ": inner dimensions differ");
        }
        int n = data.length;
        int p = other.cols;
        double[][] out = new double[n][p];
        for (int i = 0; i < n; i++) {
            double[] rowI = data[i];
            double[] outI = out[i];
            for (int k = 0; k < cols; k++) {
                double a = rowI[k];
                if (a == 0.0) {
                    continue;
                }
                double[] rowK = other.data[k];
                for (int j = 0; j < p; j++) {
                    outI[j] += a * rowK[j];
                }
            }
        }
        return new Matrix(out, p);
    }

    /** Element-wise add. A (1, cols) operand broadcasts down every row. */
    public Matrix add(Matrix other) {
        return combine(other, "add", (a, b) -> a + b);
    }

    public Matrix subtract(Matrix other) {
        return combine(other, "subtract", (a, b) -> a - b);
    }

    /** Element-wise (Hadamard) product. */
    public Matrix hadamard(Matrix other) {
        return combine(other, "hadamard", (a, b) -> a * b);
    }

    public Matrix scale(double factor) {
        return map(v -> v * factor);
    }

    public Matrix transpose() {
        double[][] out = new double[cols][data.length];
        for (int r = 0; r < data.length; r++) {
            for (int c = 0; c < cols; c++) {
                out[c][r] = data[r][c];
            }
        }
        return new Matrix(out, data.length);
    }

    /** Applies {@code f} to every element. */
    public Matrix map(DoubleUnaryOperator f) {
        Objects.requireNonNull(f, "f must not be null");
        double[][] out = new double[data.length][cols];
        for (int r = 0; r < data.length; r++) {
            for (int c = 0; c < cols; c++) {
                out[r][c] = f.applyAsDouble(data[r][c]);
            }
        }
        return new Matrix(out, cols);
    }

    /** Column-wise sum collapsed to a single (1, cols) row — used for bias gradients. */
    public Matrix sumRows() {
        double[][] out = new double[1][cols];
        for (double[] row : data) {
            for (int c = 0; c < cols; c++) {
                out[0][c] += row[c];
            }
        }
        return new Matrix(out, cols);
    }

    public double sum() {
        double total = 0.0;
        for (double[] row : data) {
            for (double v : row) {
                total += v;
            }
        }
        return total;
    }

    /** Index of the largest value in row {@code i}. */
    public int argMaxInRow(int i) {
        double[] row = data[i];
        if (row.length == 0) {
            throw new IllegalStateException("row " + i + " has no columns");
        }
        int best = 0;
        for (int c = 1; c < row.length; c++) {
            if (row[c] > row[best]) {
                best = c;
            }
        }
        return best;
    }

    /** False if any element is NaN or infinite — the training loop uses this as a divergence guard. */
    public boolean isFinite() {
        for (double[] row : data) {
            for (double v : row) {
                if (!Double.isFinite(v)) {
                    return false;
                }
            }
        }
        return true;
    }

    /** Shape as "(rows x cols)", for error messages and the UI. */
    public String shape() {
        return "(" + data.length + " x " + cols + ")";
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Matrix").append(shape());
        for (double[] row : data) {
            sb.append('\n');
            for (int c = 0; c < cols; c++) {
                sb.append(String.format("%9.4f", row[c]));
            }
        }
        return sb.toString();
    }

    private interface Op {
        double apply(double a, double b);
    }

    private Matrix combine(Matrix other, String name, Op op) {
        Objects.requireNonNull(other, "other must not be null");
        boolean broadcast = other.data.length == 1 && data.length != 1;
        if (other.cols != cols || (!broadcast && other.data.length != data.length)) {
            throw new IllegalArgumentException(
                    "cannot " + name + " " + shape() + " and " + other.shape()
                            + ": shapes must match, or the second operand must be a single row");
        }
        double[][] out = new double[data.length][cols];
        for (int r = 0; r < data.length; r++) {
            double[] rhs = other.data[broadcast ? 0 : r];
            for (int c = 0; c < cols; c++) {
                out[r][c] = op.apply(data[r][c], rhs[c]);
            }
        }
        return new Matrix(out, cols);
    }

    private static void requireNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must not be negative but was " + value);
        }
    }
}
