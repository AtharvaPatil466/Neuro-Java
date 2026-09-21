package com.neurojava.data;

import com.neurojava.core.Matrix;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Standalone self-check for the data package — no JUnit, just {@code java com.neurojava.data.DataCheck}.
 * Every check prints a PASS line so the output is demo-ready.
 */
public final class DataCheck {

    private static final double MOMENT_TOLERANCE = 1e-9;
    private static final double EXACT_TOLERANCE = 1e-12;
    private static final int IRIS_ROWS = 150;
    private static final int PER_CLASS = 50;
    private static final long SEED = 42L;

    private DataCheck() {
    }

    public static void main(String[] args) {
        XorDataset xor = new XorDataset();
        IrisDataset iris = new IrisDataset();

        checkXorShapesAndValues(xor);
        checkXorNotLinearlySeparable(xor);
        checkIrisShapesAndLabels(iris);
        checkIrisStandardised(iris);
        checkStandardizeRoundTrip(iris);
        checkIrisSplit(iris);
        checkXorFullSplit(xor);
        checkInvalidFractionsRejected(iris);

        System.out.println("ALL DATA CHECKS PASSED");
    }

    // 1a. XOR shapes and every input/target pair.
    private static void checkXorShapesAndValues(XorDataset xor) {
        double[][] expectedInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] expectedTargets = {{0}, {1}, {1}, {0}};

        Matrix inputs = xor.inputs();
        Matrix targets = xor.targets();
        check(inputs.rows() == 4 && inputs.cols() == 2, "XOR inputs should be 4x2 but were " + inputs.shape());
        check(targets.rows() == 4 && targets.cols() == 1, "XOR targets should be 4x1 but were " + targets.shape());
        check(xor.inputSize() == 2 && xor.outputSize() == 1 && xor.isBinary(),
                "XOR should be 2 in / 1 out / binary but was " + xor.inputSize() + " in / "
                        + xor.outputSize() + " out / binary=" + xor.isBinary());
        check(inputs == xor.inputs() && targets == xor.targets(),
                "XOR matrices should be built once, not rebuilt on every call");

        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 2; c++) {
                check(inputs.get(r, c) == expectedInputs[r][c], "XOR input row " + r + " col " + c
                        + " should be " + expectedInputs[r][c] + " but was " + inputs.get(r, c));
            }
            check(targets.get(r, 0) == expectedTargets[r][0], "XOR target row " + r + " (inputs "
                    + inputs.get(r, 0) + "," + inputs.get(r, 1) + ") should be " + expectedTargets[r][0]
                    + " but was " + targets.get(r, 0));
        }
        System.out.println("PASS 1  XOR: 4x2 inputs, 4x1 targets, all four truth-table rows correct");
    }

    // 1b. No single linear threshold on either feature separates the XOR labels.
    private static void checkXorNotLinearlySeparable(XorDataset xor) {
        Matrix inputs = xor.inputs();
        Matrix targets = xor.targets();
        for (int feature = 0; feature < inputs.cols(); feature++) {
            for (double threshold = -0.5; threshold <= 1.5; threshold += 0.5) {
                check(!separates(inputs, targets, feature, threshold, true)
                                && !separates(inputs, targets, feature, threshold, false),
                        "XOR was separated by feature " + feature + " at threshold " + threshold
                                + " — the data is wrong, XOR must need a hidden layer");
            }
        }
        System.out.println("PASS 2  XOR is NOT linearly separable on either feature at any threshold "
                + "-> a hidden layer is required");
    }

    private static boolean separates(Matrix inputs, Matrix targets, int feature, double threshold, boolean aboveIsOne) {
        for (int r = 0; r < inputs.rows(); r++) {
            boolean predictedOne = aboveIsOne == (inputs.get(r, feature) > threshold);
            if (predictedOne != (targets.get(r, 0) == 1.0)) {
                return false;
            }
        }
        return true;
    }

    // 2. Iris loads with the right shapes, one-hot targets and 50 samples per class.
    private static void checkIrisShapesAndLabels(IrisDataset iris) {
        Matrix inputs = iris.inputs();
        Matrix targets = iris.targets();
        check(inputs.rows() == IRIS_ROWS && inputs.cols() == 4,
                "Iris inputs should be 150x4 but were " + inputs.shape());
        check(targets.rows() == IRIS_ROWS && targets.cols() == 3,
                "Iris targets should be 150x3 but were " + targets.shape());

        int[] perClass = new int[3];
        for (int r = 0; r < targets.rows(); r++) {
            double sum = 0;
            int hot = -1;
            for (int c = 0; c < targets.cols(); c++) {
                double v = targets.get(r, c);
                check(v == 0.0 || v == 1.0,
                        "Iris target row " + r + " col " + c + " should be 0 or 1 but was " + v);
                sum += v;
                if (v == 1.0) {
                    hot = c;
                }
            }
            check(sum == 1.0, "Iris target row " + r + " should be one-hot (sum 1.0) but summed to " + sum);
            perClass[hot]++;
        }
        for (int c = 0; c < perClass.length; c++) {
            check(perClass[c] == PER_CLASS, "class " + iris.classNames()[c] + " should have "
                    + PER_CLASS + " samples but had " + perClass[c]);
        }
        System.out.println("PASS 3  Iris: 150x4 inputs, 150x3 one-hot targets, "
                + Arrays.toString(perClass) + " samples per class");
    }

    // 3. Every standardised column has mean ~0 and standard deviation ~1.
    private static void checkIrisStandardised(IrisDataset iris) {
        Matrix inputs = iris.inputs();
        for (int c = 0; c < inputs.cols(); c++) {
            double mean = 0;
            for (int r = 0; r < inputs.rows(); r++) {
                mean += inputs.get(r, c);
            }
            mean /= inputs.rows();

            double variance = 0;
            for (int r = 0; r < inputs.rows(); r++) {
                double diff = inputs.get(r, c) - mean;
                variance += diff * diff;
            }
            double sd = Math.sqrt(variance / inputs.rows());

            check(Math.abs(mean) < MOMENT_TOLERANCE, "standardised column " + iris.featureNames()[c]
                    + " should have mean ~0 but had " + mean);
            check(Math.abs(sd - 1.0) < MOMENT_TOLERANCE, "standardised column " + iris.featureNames()[c]
                    + " should have sd ~1 but had " + sd);
        }
        System.out.println("PASS 4  Every standardised Iris column has mean 0 and sd 1 (within 1e-9)");
    }

    // 4. standardize() round-trips: the raw mean row maps to all zeros.
    private static void checkStandardizeRoundTrip(IrisDataset iris) {
        Matrix zeros = iris.standardize(Matrix.rowVector(iris.featureMeans()));
        check(zeros.rows() == 1 && zeros.cols() == 4,
                "standardize should return a 1x4 row but returned " + zeros.shape());
        for (int c = 0; c < zeros.cols(); c++) {
            check(Math.abs(zeros.get(c > 0 ? 0 : 0, c)) < EXACT_TOLERANCE,
                    "standardising the mean of " + iris.featureNames()[c]
                            + " should give 0 but gave " + zeros.get(0, c));
        }
        double[] stdDevs = iris.featureStdDevs();
        for (int c = 0; c < stdDevs.length; c++) {
            check(stdDevs[c] > 0, "stored sd for " + iris.featureNames()[c] + " must be > 0 but was " + stdDevs[c]);
        }
        boolean rejected = false;
        try {
            iris.standardize(Matrix.rowVector(1.0, 2.0));
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "standardize should reject a 2-column sample for a 4-feature dataset");
        System.out.println("PASS 5  standardize(rawMeanRow) -> all zeros, and a wrong column count is rejected");
    }

    // 5. An 80/20 split is sized right, row-aligned, and a true partition of the data.
    private static void checkIrisSplit(IrisDataset iris) {
        Split split = Split.of(iris, 0.8, new Random(SEED));
        check(split.trainInputs().rows() == 120, "80% of 150 should be 120 train rows but was "
                + split.trainInputs().rows());
        check(split.valInputs().rows() == 30, "20% of 150 should be 30 validation rows but was "
                + split.valInputs().rows());
        check(split.trainTargets().rows() == 120 && split.valTargets().rows() == 30,
                "target rows must match input rows but were " + split.trainTargets().rows()
                        + " train / " + split.valTargets().rows() + " validation");
        check(split.hasValidation(), "an 80/20 split must report hasValidation() == true");

        // Each (features -> label) pair in the split must be one the dataset actually contains,
        // and the two halves together must use every original row exactly once.
        Map<String, Integer> remaining = new HashMap<>();
        for (int r = 0; r < iris.inputs().rows(); r++) {
            remaining.merge(rowKey(iris.inputs(), iris.targets(), r), 1, Integer::sum);
        }
        consume(remaining, split.trainInputs(), split.trainTargets());
        consume(remaining, split.valInputs(), split.valTargets());
        check(remaining.isEmpty(), "train + validation should use every original row exactly once, "
                + "but " + remaining.size() + " row(s) were left over or double-counted: " + remaining);
        System.out.println("PASS 6  Split.of(iris, 0.8, seed 42): 120 train + 30 validation, "
                + "labels still aligned with their features, halves disjoint and complete");
    }

    private static void consume(Map<String, Integer> remaining, Matrix inputs, Matrix targets) {
        for (int r = 0; r < inputs.rows(); r++) {
            String key = rowKey(inputs, targets, r);
            Integer count = remaining.get(key);
            check(count != null, "split row " + r + " (" + key + ") is not an original dataset row, "
                    + "or appears in both halves");
            if (count == 1) {
                remaining.remove(key);
            } else {
                remaining.put(key, count - 1);
            }
        }
    }

    private static String rowKey(Matrix inputs, Matrix targets, int row) {
        StringBuilder key = new StringBuilder();
        for (int c = 0; c < inputs.cols(); c++) {
            key.append(inputs.get(row, c)).append(',');
        }
        key.append("->");
        for (int c = 0; c < targets.cols(); c++) {
            key.append(targets.get(row, c)).append(',');
        }
        return key.toString();
    }

    // 6. A fraction of 1.0 trains on everything and leaves an empty-but-shaped validation set.
    private static void checkXorFullSplit(XorDataset xor) {
        Split split = Split.of(xor, 1.0, new Random(SEED));
        check(split.trainInputs().rows() == 4 && split.trainTargets().rows() == 4,
                "a 1.0 split of XOR should keep all 4 rows for training but kept "
                        + split.trainInputs().rows());
        check(split.valInputs().rows() == 0 && split.valTargets().rows() == 0,
                "a 1.0 split should leave 0 validation rows but left " + split.valInputs().rows());
        check(split.valInputs().cols() == 2 && split.valTargets().cols() == 1,
                "empty validation matrices must keep their column counts but were "
                        + split.valInputs().shape() + " / " + split.valTargets().shape());
        check(!split.hasValidation(), "a 1.0 split must report hasValidation() == false");
        System.out.println("PASS 7  Split.of(xor, 1.0): 4 train rows, empty (0x2 / 0x1) validation, "
                + "hasValidation() false");
    }

    // 7. Out-of-range fractions are rejected.
    private static void checkInvalidFractionsRejected(Dataset dataset) {
        for (double bad : new double[]{0.0, 1.5}) {
            boolean rejected = false;
            try {
                Split.of(dataset, bad, new Random(SEED));
            } catch (IllegalArgumentException expected) {
                rejected = true;
            }
            check(rejected, "trainFraction " + bad + " should have been rejected but was accepted");
        }
        System.out.println("PASS 8  Invalid trainFraction values (0.0 and 1.5) are both rejected");
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException("CHECK FAILED: " + message);
        }
    }
}
