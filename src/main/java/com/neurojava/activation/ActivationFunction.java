package com.neurojava.activation;

import com.neurojava.core.Matrix;

/** A layer's non-linearity, plus the derivative backpropagation needs. */
public interface ActivationFunction {

    /** Display name, e.g. "ReLU". */
    String name();

    /** Human-readable formula shown in the UI, e.g. "f(x) = max(0, x)". */
    String formula();

    /** Forward: A = f(Z). */
    Matrix apply(Matrix z);

    /**
     * Backward: returns dL/dZ given dL/dA.
     *
     * @param z pre-activation values cached from the forward pass
     * @param a post-activation values cached from the forward pass (f(z))
     * @param gradA incoming gradient dL/dA, same shape as z
     */
    Matrix gradient(Matrix z, Matrix a, Matrix gradA);
}
