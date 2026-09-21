package com.neurojava.data;

import com.neurojava.core.Matrix;

import java.util.Objects;
import java.util.Random;

/** A shuffled train/validation partition of a dataset. */
public record Split(Matrix trainInputs, Matrix trainTargets, Matrix valInputs, Matrix valTargets) {

    private static final int MIN_TRAIN_ROWS = 1;

    /**
     * Shuffles and splits, keeping {@code trainFraction} of the rows for training.
     * A fraction of 1.0 leaves the validation matrices empty (zero rows).
     */
    public static Split of(Dataset dataset, double trainFraction, Random rng) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(rng, "rng must not be null");
        if (!(trainFraction > 0.0) || trainFraction > 1.0) {
            throw new IllegalArgumentException(
                    "trainFraction must be in (0, 1] but was " + trainFraction);
        }

        Matrix inputs = dataset.inputs();
        Matrix targets = dataset.targets();
        int total = inputs.rows();
        if (total != targets.rows()) {
            throw new IllegalArgumentException("dataset " + dataset.name() + " has " + total
                    + " input rows but " + targets.rows() + " target rows");
        }
        if (total == 0) {
            throw new IllegalArgumentException("dataset " + dataset.name() + " has no rows to split");
        }

        int[] order = shuffledIndices(total, rng);
        int trainCount = Math.min(total, Math.max(MIN_TRAIN_ROWS, (int) Math.round(total * trainFraction)));

        int[] trainIdx = new int[trainCount];
        int[] valIdx = new int[total - trainCount];
        System.arraycopy(order, 0, trainIdx, 0, trainCount);
        System.arraycopy(order, trainCount, valIdx, 0, valIdx.length);

        // selectRows with an empty index array yields a 0-row matrix that keeps the column count,
        // which is exactly Matrix.zeros(0, cols) — the empty-validation representation.
        return new Split(
                inputs.selectRows(trainIdx),
                targets.selectRows(trainIdx),
                inputs.selectRows(valIdx),
                targets.selectRows(valIdx));
    }

    /** Fisher-Yates shuffle of 0..n-1 using the supplied source of randomness. */
    private static int[] shuffledIndices(int n, Random rng) {
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int swap = order[i];
            order[i] = order[j];
            order[j] = swap;
        }
        return order;
    }

    public boolean hasValidation() {
        return valInputs.rows() > 0;
    }
}
