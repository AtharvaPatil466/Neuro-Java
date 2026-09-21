package com.neurojava.data;

import com.neurojava.core.Matrix;

/** A labelled dataset the network can train on. */
public interface Dataset {

    String name();

    /** Feature matrix, one row per sample, already scaled for training. */
    Matrix inputs();

    /** Targets: one column for binary problems, one-hot columns for multiclass. */
    Matrix targets();

    String[] featureNames();

    /** Class labels, indexed to match the one-hot target columns. */
    String[] classNames();

    int inputSize();

    int outputSize();

    /** True when a single sigmoid output is the right head (XOR); false for one-hot softmax (Iris). */
    boolean isBinary();

    /**
     * Applies the same feature scaling that produced {@link #inputs()} to raw, user-entered
     * samples, so a prediction typed into the UI is comparable with the training data.
     * Datasets that need no scaling return the argument unchanged.
     */
    Matrix standardize(Matrix rawSamples);
}
