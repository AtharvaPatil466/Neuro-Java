package com.neurojava.activation;

import com.neurojava.core.Matrix;

/** Rectified linear unit — the default hidden-layer non-linearity. */
public final class ReLU implements ActivationFunction {

    @Override
    public String name() {
        return "ReLU";
    }

    @Override
    public String formula() {
        return "f(x) = max(0, x)";
    }

    @Override
    public Matrix apply(Matrix z) {
        return z.map(v -> Math.max(0.0, v));
    }

    /**
     * dL/dZ = dL/dA * 1{z &gt; 0}. ReLU is not differentiable at exactly 0; by the usual
     * convention we take the sub-gradient 0 there.
     */
    @Override
    public Matrix gradient(Matrix z, Matrix a, Matrix gradA) {
        return gradA.hadamard(z.map(v -> v > 0.0 ? 1.0 : 0.0));
    }
}
