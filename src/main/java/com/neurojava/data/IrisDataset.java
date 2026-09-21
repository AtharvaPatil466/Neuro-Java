package com.neurojava.data;

import com.neurojava.core.Matrix;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Fisher's Iris dataset: 150 flowers, four measurements each, three species.
 * Features are z-score standardised column-wise so every input reaches the network on the
 * same scale; the per-column mean and standard deviation are kept so a raw sample typed into
 * the UI can be put on that same scale by {@link #standardize(Matrix)}.
 */
public final class IrisDataset implements Dataset {

    private static final String RESOURCE = "/data/iris.csv";
    private static final int FEATURE_COUNT = 4;
    private static final int CLASS_COUNT = 3;
    private static final int CSV_COLUMNS = FEATURE_COUNT + 1;
    /** Substituted for a zero standard deviation so standardising never divides by zero. */
    private static final double SAFE_STD_DEV = 1.0;

    private static final String[] FEATURE_NAMES =
            {"Sepal Length", "Sepal Width", "Petal Length", "Petal Width"};
    private static final String[] CLASS_NAMES = {"Setosa", "Versicolor", "Virginica"};
    private static final String[] SPECIES_LABELS = {"setosa", "versicolor", "virginica"};

    private final Matrix inputs;
    private final Matrix targets;
    private final double[] means;
    private final double[] stdDevs;

    public IrisDataset() {
        List<double[]> rawRows = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        load(rawRows, labels);

        double[][] raw = rawRows.toArray(new double[0][]);
        this.means = columnMeans(raw);
        this.stdDevs = columnStdDevs(raw, means);
        this.inputs = new Matrix(applyScaling(raw));
        this.targets = new Matrix(oneHot(labels));
    }

    // ---------------------------------------------------------------- loading

    private static void load(List<double[]> rawRows, List<Integer> labels) {
        InputStream stream = IrisDataset.class.getResourceAsStream(RESOURCE);
        if (stream == null) {
            throw new IllegalStateException(
                    "Iris data file not found on the classpath at " + RESOURCE
                            + " — it must be packaged under src/main/resources" + RESOURCE);
        }
        try (BufferedReader reader =
                     new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            int lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (lineNumber == 1 || line.isBlank()) {
                    continue; // header row, and any trailing newline at end of file
                }
                parseRow(line, lineNumber, rawRows, labels);
            }
        } catch (IOException e) {
            throw new IllegalStateException("failed to read " + RESOURCE + ": " + e.getMessage(), e);
        }
        if (rawRows.isEmpty()) {
            throw new IllegalStateException(RESOURCE + " contained a header but no data rows");
        }
    }

    private static void parseRow(String line, int lineNumber, List<double[]> rawRows, List<Integer> labels) {
        String[] parts = line.split(",", -1);
        if (parts.length != CSV_COLUMNS) {
            throw new IllegalArgumentException("malformed row at line " + lineNumber + ": expected "
                    + CSV_COLUMNS + " comma-separated values but found " + parts.length + " in: " + line);
        }
        double[] features = new double[FEATURE_COUNT];
        for (int c = 0; c < FEATURE_COUNT; c++) {
            String value = parts[c].trim();
            try {
                features[c] = Double.parseDouble(value);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("malformed row at line " + lineNumber + ": column "
                        + (c + 1) + " (\"" + value + "\") is not a number, in: " + line, e);
            }
        }
        rawRows.add(features);
        labels.add(labelIndex(parts[FEATURE_COUNT].trim(), lineNumber, line));
    }

    private static int labelIndex(String species, int lineNumber, String line) {
        for (int i = 0; i < SPECIES_LABELS.length; i++) {
            if (SPECIES_LABELS[i].equalsIgnoreCase(species)) {
                return i;
            }
        }
        throw new IllegalArgumentException("unknown species \"" + species + "\" at line " + lineNumber
                + ": expected one of " + String.join(", ", SPECIES_LABELS) + ", in: " + line);
    }

    // ------------------------------------------------------------- statistics

    private static double[] columnMeans(double[][] raw) {
        double[] means = new double[FEATURE_COUNT];
        for (double[] row : raw) {
            for (int c = 0; c < FEATURE_COUNT; c++) {
                means[c] += row[c];
            }
        }
        for (int c = 0; c < FEATURE_COUNT; c++) {
            means[c] /= raw.length;
        }
        return means;
    }

    private static double[] columnStdDevs(double[][] raw, double[] means) {
        double[] stdDevs = new double[FEATURE_COUNT];
        for (double[] row : raw) {
            for (int c = 0; c < FEATURE_COUNT; c++) {
                double diff = row[c] - means[c];
                stdDevs[c] += diff * diff;
            }
        }
        for (int c = 0; c < FEATURE_COUNT; c++) {
            double sd = Math.sqrt(stdDevs[c] / raw.length);
            stdDevs[c] = sd == 0.0 ? SAFE_STD_DEV : sd;
        }
        return stdDevs;
    }

    private double[][] applyScaling(double[][] raw) {
        double[][] scaled = new double[raw.length][FEATURE_COUNT];
        for (int r = 0; r < raw.length; r++) {
            for (int c = 0; c < FEATURE_COUNT; c++) {
                scaled[r][c] = (raw[r][c] - means[c]) / stdDevs[c];
            }
        }
        return scaled;
    }

    private static double[][] oneHot(List<Integer> labels) {
        double[][] out = new double[labels.size()][CLASS_COUNT];
        for (int r = 0; r < labels.size(); r++) {
            out[r][labels.get(r)] = 1.0;
        }
        return out;
    }

    // ---------------------------------------------------------------- dataset

    @Override
    public String name() {
        return "Iris";
    }

    @Override
    public Matrix inputs() {
        return inputs;
    }

    @Override
    public Matrix targets() {
        return targets;
    }

    @Override
    public String[] featureNames() {
        return FEATURE_NAMES.clone();
    }

    @Override
    public String[] classNames() {
        return CLASS_NAMES.clone();
    }

    @Override
    public int inputSize() {
        return FEATURE_COUNT;
    }

    @Override
    public int outputSize() {
        return CLASS_COUNT;
    }

    @Override
    public boolean isBinary() {
        return false;
    }

    @Override
    public Matrix standardize(Matrix rawSamples) {
        if (rawSamples == null) {
            throw new IllegalArgumentException("rawSamples must not be null");
        }
        if (rawSamples.cols() != FEATURE_COUNT) {
            throw new IllegalArgumentException("Iris expects " + FEATURE_COUNT
                    + " features per sample but got " + rawSamples.cols() + " (shape " + rawSamples.shape() + ")");
        }
        return new Matrix(applyScaling(rawSamples.toArray()));
    }

    /** Per-column means of the raw measurements, so the UI can offer a typical sample. */
    public double[] featureMeans() {
        return means.clone();
    }

    /** Per-column standard deviations of the raw measurements (never zero). */
    public double[] featureStdDevs() {
        return stdDevs.clone();
    }
}
