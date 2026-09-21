package com.neurojava.nn;

import com.neurojava.core.Matrix;

/** One trainable stage of the network. */
public interface Layer {

    /** Forward pass; caches whatever {@link #backward} will need. */
    Matrix forward(Matrix input);

    /**
     * Backward pass: accumulates this layer's parameter gradients from {@code gradOutput}
     * (dL/dOutput) and returns dL/dInput for the previous layer.
     */
    Matrix backward(Matrix gradOutput);

    /** Gradient-descent step: parameter -= learningRate * gradient. */
    void updateWeights(double learningRate);

    int inputSize();

    int outputSize();
}
