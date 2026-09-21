package com.neurojava.viz;

import com.neurojava.data.Dataset;

/**
 * One fully parsed, already validated configuration from the control panel.
 *
 * @param hiddenSizes neurons per hidden layer, e.g. {@code {8, 6}}; may be empty
 * @param hiddenActivation "ReLU" or "Sigmoid" — the output head is chosen by the dataset, not here
 */
record TrainingSpec(Dataset dataset,
                    int[] hiddenSizes,
                    String hiddenActivation,
                    double learningRate,
                    int epochs,
                    int batchSize,
                    long seed,
                    double trainFraction) {
}
