package com.neurojava.data;

import com.neurojava.core.Matrix;

/**
 * The canonical non-linearly-separable toy problem: output is 1 when exactly one input is 1.
 * No single straight line splits the two classes, which is precisely why the network needs a
 * hidden layer — see {@link DataCheck} for the check that proves it.
 */
public final class XorDataset implements Dataset {

    private static final Matrix INPUTS = new Matrix(new double[][]{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
    });

    private static final Matrix TARGETS = new Matrix(new double[][]{
            {0},
            {1},
            {1},
            {0}
    });

    private static final String[] FEATURE_NAMES = {"Input A", "Input B"};
    private static final String[] CLASS_NAMES = {"0", "1"};

    @Override
    public String name() {
        return "XOR";
    }

    @Override
    public Matrix inputs() {
        return INPUTS;
    }

    @Override
    public Matrix targets() {
        return TARGETS;
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
        return 2;
    }

    @Override
    public int outputSize() {
        return 1;
    }

    @Override
    public boolean isBinary() {
        return true;
    }

    @Override
    public Matrix standardize(Matrix rawSamples) {
        // XOR inputs are already 0/1, so no scaling is needed: the raw sample IS the training scale.
        return rawSamples;
    }
}
