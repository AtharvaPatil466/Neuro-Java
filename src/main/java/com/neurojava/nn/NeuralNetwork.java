package com.neurojava.nn;

import com.neurojava.core.Matrix;
import com.neurojava.loss.LossFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Ordered stack of layers plus the loss that drives backpropagation. */
public final class NeuralNetwork {

    private final LossFunction loss;
    private final List<Layer> layers = new ArrayList<>();

    public NeuralNetwork(LossFunction loss) {
        this.loss = Objects.requireNonNull(loss, "loss must not be null");
    }

    /** Appends a layer. Returns {@code this} so layers can be chained. */
    public NeuralNetwork add(Layer layer) {
        Objects.requireNonNull(layer, "layer must not be null");
        if (!layers.isEmpty()) {
            int previousOutput = layers.get(layers.size() - 1).outputSize();
            if (layer.inputSize() != previousOutput) {
                throw new IllegalArgumentException(
                        "layer expects " + layer.inputSize() + " inputs but the previous layer outputs "
                                + previousOutput);
            }
        }
        layers.add(layer);
        return this;
    }

    /** Runs every layer in order and returns the output activations. */
    public Matrix forward(Matrix input) {
        Objects.requireNonNull(input, "input must not be null");
        requireLayers();
        Matrix activations = input;
        for (Layer layer : layers) {
            activations = layer.forward(activations);
        }
        return activations;
    }

    /**
     * Seeds the chain rule from dLoss/dPredicted and propagates it back through every layer,
     * leaving each layer holding its parameter gradients.
     *
     * @return the scalar loss for this batch
     */
    public double backward(Matrix predicted, Matrix target) {
        Objects.requireNonNull(predicted, "predicted must not be null");
        Objects.requireNonNull(target, "target must not be null");
        requireLayers();
        double value = loss.value(predicted, target);
        Matrix gradient = loss.gradient(predicted, target);
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers.get(i).backward(gradient);
        }
        return value;
    }

    public void updateWeights(double learningRate) {
        for (Layer layer : layers) {
            layer.updateWeights(learningRate);
        }
    }

    /** Forward pass without touching cached-gradient state semantics; used after training. */
    public Matrix predict(Matrix input) {
        return forward(input);
    }

    public List<Layer> layers() {
        return Collections.unmodifiableList(layers);
    }

    public LossFunction loss() {
        return loss;
    }

    /** Neuron counts per layer including the input, e.g. {@code [2, 4, 1]}. */
    public int[] architecture() {
        requireLayers();
        int[] sizes = new int[layers.size() + 1];
        sizes[0] = layers.get(0).inputSize();
        for (int i = 0; i < layers.size(); i++) {
            sizes[i + 1] = layers.get(i).outputSize();
        }
        return sizes;
    }

    private void requireLayers() {
        if (layers.isEmpty()) {
            throw new IllegalStateException("network has no layers; call add(...) first");
        }
    }
}
