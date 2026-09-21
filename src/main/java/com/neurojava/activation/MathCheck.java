package com.neurojava.activation;

import com.neurojava.core.Matrix;
import com.neurojava.loss.BinaryCrossEntropy;
import com.neurojava.loss.CategoricalCrossEntropy;
import com.neurojava.loss.LossFunction;
import com.neurojava.loss.MeanSquaredError;
import java.util.Random;
import java.util.function.ToDoubleFunction;

/**
 * Standalone gradient checker: no JUnit, just {@code java com.neurojava.activation.MathCheck}.
 *
 * <p>Every analytic derivative in the activation and loss packages is compared against a central
 * finite difference of the very function it claims to differentiate. If backprop ever disagrees
 * with the forward pass, this is where it shows up.
 */
public final class MathCheck {

    /** Step for central differences: small enough for h^2 truncation, big enough to dodge roundoff. */
    private static final double H = 1e-6;
    private static final double REL_TOL = 1e-6;
    /** Fallback when both gradients are ~0 and a relative comparison is meaningless. */
    private static final double ABS_TOL = 1e-8;
    private static final long SEED = 20240517L;

    private MathCheck() {
    }

    public static void main(String[] args) {
        Random rng = new Random(SEED);
        checkActivations(rng);
        checkLosses(rng);
        checkSoftmaxThenCrossEntropy(rng);
        checkSoftmaxStability();
        checkSigmoidStability();
        System.out.println("PASS  all 5 checks green");
    }

    // ---------------------------------------------------------------- activations

    private static void checkActivations(Random rng) {
        // ReLU: hand-picked so the batch spans strongly negative, near-zero-negative,
        // near-zero-positive and strongly positive inputs (never exactly 0, where the kink lives).
        Matrix reluZ = new Matrix(new double[][]{
                {-3.0, -1e-3, 1e-3, 2.5},
                {0.75, -0.5, 4.2, -2.25}
        });
        checkActivation(new ReLU(), reluZ, Matrix.random(2, 4, 1.5, rng), "ReLU");
        checkActivation(new Sigmoid(), Matrix.random(3, 4, 3.0, rng), Matrix.random(3, 4, 1.5, rng), "Sigmoid");
        checkActivation(new Softmax(), Matrix.random(3, 4, 2.0, rng), Matrix.random(3, 4, 1.5, rng), "Softmax");
    }

    /**
     * Uses the scalar objective L = sum(W * f(Z)) for a fixed random W, so dL/dA is exactly W
     * and the activation's own {@code gradient} must reproduce dL/dZ.
     */
    private static void checkActivation(ActivationFunction f, Matrix z, Matrix gradA, String label) {
        Matrix analytic = f.gradient(z, f.apply(z), gradA);
        Matrix numeric = numericalGradient(z, perturbed -> f.apply(perturbed).hadamard(gradA).sum());
        compare(analytic, numeric, label + ".gradient");
        System.out.println("PASS  " + label + " dL/dZ matches central differences on " + z.shape()
                + "  [" + f.formula() + "]");
    }

    // ---------------------------------------------------------------- losses

    private static void checkLosses(Random rng) {
        checkLoss(new MeanSquaredError(),
                Matrix.random(4, 3, 2.0, rng), Matrix.random(4, 3, 2.0, rng));
        checkLoss(new BinaryCrossEntropy(),
                new Sigmoid().apply(Matrix.random(4, 2, 2.0, rng)), binaryTargets(4, 2, rng));
        checkLoss(new CategoricalCrossEntropy(),
                new Softmax().apply(Matrix.random(4, 3, 2.0, rng)), oneHotTargets(4, 3, rng));
    }

    private static void checkLoss(LossFunction loss, Matrix predicted, Matrix target) {
        Matrix analytic = loss.gradient(predicted, target);
        Matrix numeric = numericalGradient(predicted, perturbed -> loss.value(perturbed, target));
        compare(analytic, numeric, loss.name() + ".gradient");
        System.out.println("PASS  " + loss.name() + " dL/dP matches central differences of its own value()"
                + "  (loss = " + String.format("%.6f", loss.value(predicted, target)) + ")");
    }

    // ---------------------------------------------------------------- composed path

    /**
     * The check that earns the un-fused softmax: perturb the pre-activation Z, push it through
     * softmax and then categorical cross entropy, and confirm the chained analytic gradient
     * (loss gradient -> softmax Jacobian-vector product) equals the finite difference of the
     * composed scalar. A fused (a - y) shortcut would also pass end-to-end here, but only this
     * form keeps softmax honest as a standalone activation.
     */
    private static void checkSoftmaxThenCrossEntropy(Random rng) {
        Softmax softmax = new Softmax();
        CategoricalCrossEntropy loss = new CategoricalCrossEntropy();
        Matrix z = Matrix.random(5, 4, 2.0, rng);
        Matrix target = oneHotTargets(5, 4, rng);

        Matrix a = softmax.apply(z);
        Matrix analytic = softmax.gradient(z, a, loss.gradient(a, target));
        Matrix numeric = numericalGradient(z, perturbed -> loss.value(softmax.apply(perturbed), target));
        compare(analytic, numeric, "Softmax -> CategoricalCrossEntropy");
        System.out.println("PASS  composed Softmax -> Categorical Cross Entropy dL/dZ matches central differences"
                + " on " + z.shape());
    }

    // ---------------------------------------------------------------- numerical stability

    private static void checkSoftmaxStability() {
        Softmax softmax = new Softmax();
        Matrix extreme = new Matrix(new double[][]{
                {1000.0, -1000.0, 0.0},
                {-1000.0, -1000.0, -1000.0},
                {0.1, 0.2, 0.3}
        });
        Matrix a = softmax.apply(extreme);
        check(a.isFinite(), "softmax produced a non-finite value for extreme inputs: " + a);
        for (int r = 0; r < a.rows(); r++) {
            double rowSum = a.row(r).sum();
            check(Math.abs(rowSum - 1.0) <= 1e-12,
                    "softmax row " + r + " sums to " + rowSum + ", expected 1.0");
        }
        System.out.println("PASS  Softmax rows sum to 1 and stay finite for [1000, -1000, 0]");
    }

    private static void checkSigmoidStability() {
        Sigmoid sigmoid = new Sigmoid();
        Matrix a = sigmoid.apply(Matrix.rowVector(800.0, -800.0, 0.0));
        check(a.isFinite(), "sigmoid produced a non-finite value at +/-800: " + a);
        check(a.get(0, 0) > 0.0 && a.get(0, 0) <= 1.0,
                "sigmoid(800) should be in (0, 1] but was " + a.get(0, 0));
        check(a.get(0, 1) >= 0.0 && a.get(0, 1) < 1e-300,
                "sigmoid(-800) should underflow towards 0 but was " + a.get(0, 1));
        check(Math.abs(a.get(0, 2) - 0.5) <= 1e-15,
                "sigmoid(0) should be 0.5 but was " + a.get(0, 2));
        System.out.println("PASS  Sigmoid finite and NaN-free at x = +/-800 (got "
                + a.get(0, 0) + " and " + a.get(0, 1) + ")");
    }

    // ---------------------------------------------------------------- helpers

    /** Central differences: (f(x + h) - f(x - h)) / 2h, one element at a time. */
    private static Matrix numericalGradient(Matrix x, ToDoubleFunction<Matrix> objective) {
        double[][] out = new double[x.rows()][x.cols()];
        for (int r = 0; r < x.rows(); r++) {
            for (int c = 0; c < x.cols(); c++) {
                double[][] plus = x.toArray();
                double[][] minus = x.toArray();
                plus[r][c] += H;
                minus[r][c] -= H;
                out[r][c] = (objective.applyAsDouble(new Matrix(plus))
                        - objective.applyAsDouble(new Matrix(minus))) / (2.0 * H);
            }
        }
        return new Matrix(out);
    }

    private static void compare(Matrix analytic, Matrix numeric, String label) {
        check(analytic.rows() == numeric.rows() && analytic.cols() == numeric.cols(),
                label + ": analytic " + analytic.shape() + " but numeric " + numeric.shape());
        check(analytic.isFinite(), label + ": analytic gradient is not finite: " + analytic);
        for (int r = 0; r < analytic.rows(); r++) {
            for (int c = 0; c < analytic.cols(); c++) {
                double an = analytic.get(r, c);
                double nu = numeric.get(r, c);
                double tol = Math.max(ABS_TOL, REL_TOL * Math.max(Math.abs(an), Math.abs(nu)));
                check(Math.abs(an - nu) <= tol,
                        label + " at [" + r + ", " + c + "]: analytic " + an + " vs numeric " + nu
                                + " (|diff| " + Math.abs(an - nu) + " > tol " + tol + ")");
            }
        }
    }

    /** Plain guard — not the {@code assert} keyword, which is disabled unless someone passes -ea. */
    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException("FAIL  " + message);
        }
    }

    private static Matrix binaryTargets(int rows, int cols, Random rng) {
        double[][] out = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out[r][c] = rng.nextBoolean() ? 1.0 : 0.0;
            }
        }
        return new Matrix(out);
    }

    private static Matrix oneHotTargets(int rows, int cols, Random rng) {
        double[][] out = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            out[r][rng.nextInt(cols)] = 1.0;
        }
        return new Matrix(out);
    }
}
