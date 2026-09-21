package com.neurojava.activation;

import com.neurojava.core.Matrix;

/** Logistic sigmoid, squashing any real value into (0, 1). */
public final class Sigmoid implements ActivationFunction {

    @Override
    public String name() {
        return "Sigmoid";
    }

    @Override
    public String formula() {
        return "f(x) = 1 / (1 + e^-x)";
    }

    @Override
    public Matrix apply(Matrix z) {
        return z.map(Sigmoid::sigmoid);
    }

    /** dL/dZ = dL/dA * a * (1 - a), reusing the cached forward output instead of recomputing it. */
    @Override
    public Matrix gradient(Matrix z, Matrix a, Matrix gradA) {
        return gradA.hadamard(a.map(v -> v * (1.0 - v)));
    }

    /**
     * Two algebraically identical branches, picked so the exponent is never positive:
     * exp(-x) would overflow for very negative x, exp(x) for very positive x.
     */
    private static double sigmoid(double x) {
        if (x >= 0.0) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
        double ex = Math.exp(x);
        return ex / (1.0 + ex);
    }
}
