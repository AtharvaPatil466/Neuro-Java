package com.neurojava.nn;

import com.neurojava.activation.ActivationFunction;
import com.neurojava.core.Matrix;

import java.util.Objects;
import java.util.Random;

/** Fully connected layer: {@code A = activation(X . W + b)}. */
public final class DenseLayer implements Layer {

    /** Matched against {@link ActivationFunction#name()} to pick He over Xavier init. */
    private static final String RELU_NAME = "ReLU";

    private final ActivationFunction activation;
    private final int inputSize;
    private final int outputSize;

    private Matrix weights;
    private Matrix biases;

    private Matrix lastInput;
    private Matrix lastPreActivation;
    private Matrix lastActivation;

    private Matrix gradWeights;
    private Matrix gradBiases;

    /**
     * He initialisation for ReLU, Xavier otherwise; biases start at zero.
     *
     * @param rng seeded so a run can be reproduced
     */
    public DenseLayer(int inputSize, int outputSize, ActivationFunction activation, Random rng) {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be positive but was " + inputSize);
        }
        if (outputSize <= 0) {
            throw new IllegalArgumentException("outputSize must be positive but was " + outputSize);
        }
        this.activation = Objects.requireNonNull(activation, "activation must not be null");
        Objects.requireNonNull(rng, "rng must not be null");
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        double scale = RELU_NAME.equals(activation.name())
                ? Math.sqrt(6.0 / inputSize)
                : Math.sqrt(6.0 / (inputSize + outputSize));
        this.weights = Matrix.random(inputSize, outputSize, scale, rng);
        this.biases = Matrix.zeros(1, outputSize);
    }

    @Override
    public Matrix forward(Matrix input) {
        Objects.requireNonNull(input, "input must not be null");
        if (input.cols() != inputSize) {
            throw new IllegalArgumentException(
                    "layer expects " + inputSize + " input features but got " + input.shape());
        }
        lastInput = input;
        lastPreActivation = input.multiply(weights).add(biases);
        lastActivation = activation.apply(lastPreActivation);
        return lastActivation;
    }

    @Override
    public Matrix backward(Matrix gradOutput) {
        Objects.requireNonNull(gradOutput, "gradOutput must not be null");
        if (lastInput == null) {
            throw new IllegalStateException("backward called before any forward pass");
        }
        if (gradOutput.rows() != lastActivation.rows() || gradOutput.cols() != outputSize) {
            throw new IllegalArgumentException(
                    "gradOutput " + gradOutput.shape() + " does not match this layer's output "
                            + lastActivation.shape());
        }
        Matrix gradPreActivation = activation.gradient(lastPreActivation, lastActivation, gradOutput);
        gradWeights = lastInput.transpose().multiply(gradPreActivation);
        gradBiases = gradPreActivation.sumRows();
        return gradPreActivation.multiply(weights.transpose());
    }

    @Override
    public void updateWeights(double learningRate) {
        if (gradWeights == null) {
            throw new IllegalStateException("updateWeights called before backward");
        }
        weights = weights.subtract(gradWeights.scale(learningRate));
        biases = biases.subtract(gradBiases.scale(learningRate));
        gradWeights = null;
        gradBiases = null;
    }

    @Override
    public int inputSize() {
        return inputSize;
    }

    @Override
    public int outputSize() {
        return outputSize;
    }

    public ActivationFunction activation() {
        return activation;
    }

    /** Current weights (in, out). Read-only snapshot for the visualiser. */
    public Matrix weights() {
        return weights;
    }

    /** Current biases (1, out). */
    public Matrix biases() {
        return biases;
    }

    /** Activations cached by the last forward pass, or null before the first one. */
    public Matrix lastActivation() {
        return lastActivation;
    }

    /** Analytic dL/dW from the last {@link #backward}, or null after an update. For {@link GradientCheck}. */
    Matrix gradWeights() {
        return gradWeights;
    }

    /** Analytic dL/db from the last {@link #backward}, or null after an update. For {@link GradientCheck}. */
    Matrix gradBiases() {
        return gradBiases;
    }

    /** Overwrites the parameters — only {@link GradientCheck} needs this, to nudge one value at a time. */
    void setParameters(Matrix newWeights, Matrix newBiases) {
        if (newWeights.rows() != inputSize || newWeights.cols() != outputSize) {
            throw new IllegalArgumentException("weights must be (" + inputSize + " x " + outputSize
                    + ") but were " + newWeights.shape());
        }
        if (newBiases.rows() != 1 || newBiases.cols() != outputSize) {
            throw new IllegalArgumentException("biases must be (1 x " + outputSize + ") but were "
                    + newBiases.shape());
        }
        weights = newWeights;
        biases = newBiases;
    }
}
