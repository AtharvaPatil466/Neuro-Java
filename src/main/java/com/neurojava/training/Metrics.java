package com.neurojava.training;

/**
 * Snapshot emitted once per epoch.
 *
 * @param valAccuracy NaN when the run has no validation split
 */
public record Metrics(int epoch,
                      int totalEpochs,
                      double loss,
                      double trainAccuracy,
                      double valAccuracy,
                      long elapsedMillis) {
}
