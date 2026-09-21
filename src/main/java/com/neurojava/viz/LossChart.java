package com.neurojava.viz;

import com.neurojava.training.Metrics;

import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;

/**
 * Live loss curve. Accuracy rides on the same axis because both cross-entropy losses and an
 * accuracy fraction sit in roughly the same 0..1 range for these datasets.
 */
final class LossChart extends LineChart<Number, Number> {

    /** Above this many points the chart is thinned rather than left to grow with the epoch count. */
    private static final int MAX_POINTS = 500;

    private final XYChart.Series<Number, Number> lossSeries = new XYChart.Series<>();
    private final XYChart.Series<Number, Number> accuracySeries = new XYChart.Series<>();
    private int stride = 1;

    LossChart() {
        super(new NumberAxis(), new NumberAxis());
        setAnimated(false);
        setCreateSymbols(false);
        setLegendVisible(true);
        getXAxis().setLabel("Epoch");
        getYAxis().setLabel("Loss / accuracy");
        lossSeries.setName("Loss");
        accuracySeries.setName("Train accuracy");
        getData().add(lossSeries);
        getData().add(accuracySeries);
    }

    void clear() {
        lossSeries.getData().clear();
        accuracySeries.getData().clear();
        stride = 1;
    }

    /**
     * Records one epoch. {@code force} is used for the final epoch so the curve always ends on the
     * real last value even when thinning would have skipped it.
     */
    void add(Metrics metrics, boolean force) {
        if (!force && metrics.epoch() % stride != 0) {
            return;
        }
        lossSeries.getData().add(new XYChart.Data<>(metrics.epoch(), metrics.loss()));
        if (!Double.isNaN(metrics.trainAccuracy())) {
            accuracySeries.getData().add(new XYChart.Data<>(metrics.epoch(), metrics.trainAccuracy()));
        }
        if (lossSeries.getData().size() > MAX_POINTS) {
            thin();
        }
    }

    /** Drops every other point and doubles the stride, halving the series in place. */
    private void thin() {
        removeAlternate(lossSeries);
        removeAlternate(accuracySeries);
        stride *= 2;
    }

    private static void removeAlternate(XYChart.Series<Number, Number> series) {
        for (int i = series.getData().size() - 2; i > 0; i -= 2) {
            series.getData().remove(i);
        }
    }
}
