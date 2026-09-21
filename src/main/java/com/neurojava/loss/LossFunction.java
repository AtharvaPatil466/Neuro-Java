package com.neurojava.loss;

import com.neurojava.core.Matrix;

/** Objective the network minimises. */
public interface LossFunction {

    /** Display name, e.g. "Binary Cross Entropy". */
    String name();

    /** Scalar loss, averaged over the batch. */
    double value(Matrix predicted, Matrix target);

    /** dL/dPredicted, already divided by the batch size so it matches {@link #value}. */
    Matrix gradient(Matrix predicted, Matrix target);
}
