package com.neurojava.viz;

import com.neurojava.activation.ActivationFunction;
import com.neurojava.activation.ReLU;
import com.neurojava.activation.Sigmoid;
import com.neurojava.activation.Softmax;
import com.neurojava.data.Dataset;
import com.neurojava.loss.BinaryCrossEntropy;
import com.neurojava.loss.CategoricalCrossEntropy;
import com.neurojava.loss.LossFunction;
import com.neurojava.nn.DenseLayer;
import com.neurojava.nn.NeuralNetwork;

import java.util.Random;

/** Turns a {@link TrainingSpec} into a network, applying the one rule the user does not choose. */
final class NetworkFactory {

    private NetworkFactory() {
    }

    static ActivationFunction hiddenActivation(String name) {
        return "ReLU".equals(name) ? new ReLU() : new Sigmoid();
    }

    /**
     * The output head follows the dataset: a binary problem gets a single sigmoid neuron with
     * binary cross entropy, anything else gets a softmax over one-hot targets with categorical
     * cross entropy. Letting the user pick that combination would only let them pick a broken one.
     */
    static NeuralNetwork build(TrainingSpec spec) {
        Dataset dataset = spec.dataset();
        Random rng = new Random(spec.seed());
        LossFunction loss = dataset.isBinary() ? new BinaryCrossEntropy() : new CategoricalCrossEntropy();
        NeuralNetwork network = new NeuralNetwork(loss);

        int previous = dataset.inputSize();
        for (int size : spec.hiddenSizes()) {
            network.add(new DenseLayer(previous, size, hiddenActivation(spec.hiddenActivation()), rng));
            previous = size;
        }
        ActivationFunction head = dataset.isBinary() ? new Sigmoid() : new Softmax();
        network.add(new DenseLayer(previous, dataset.outputSize(), head, rng));
        return network;
    }

    /** Renders {@code [2, 4, 1]} as {@code "2 -> 4 -> 1"}. */
    static String describe(int[] architecture) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < architecture.length; i++) {
            if (i > 0) {
                sb.append(" -> ");
            }
            sb.append(architecture[i]);
        }
        return sb.toString();
    }
}
