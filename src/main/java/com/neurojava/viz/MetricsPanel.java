package com.neurojava.viz;

import com.neurojava.training.Metrics;

import javafx.geometry.Insets;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Priority;

/** The numbers a demo audience actually watches: epoch, loss, accuracy, elapsed time. */
final class MetricsPanel extends GridPane {

    private static final String EMPTY = "—";
    private static final double MILLIS_PER_SECOND = 1000.0;

    private final Label epochValue = value();
    private final Label lossValue = value();
    private final Label trainAccuracyValue = value();
    private final Label validationAccuracyValue = value();
    private final Label timeValue = value();
    private final ProgressBar progress = new ProgressBar(0);

    MetricsPanel() {
        setHgap(14);
        setVgap(6);
        setPadding(new Insets(10));
        setStyle(Theme.PANEL_STYLE);

        addPair(0, "Epoch", epochValue);
        addPair(1, "Loss", lossValue);
        addPair(2, "Train accuracy", trainAccuracyValue);
        addPair(3, "Validation accuracy", validationAccuracyValue);
        addPair(4, "Training time", timeValue);

        progress.setMaxWidth(Double.MAX_VALUE);
        GridPane.setHgrow(progress, Priority.ALWAYS);
        add(progress, 0, 5, 2, 1);
        reset();
    }

    void reset() {
        epochValue.setText(EMPTY);
        lossValue.setText(EMPTY);
        trainAccuracyValue.setText(EMPTY);
        validationAccuracyValue.setText(EMPTY);
        timeValue.setText(EMPTY);
        progress.setProgress(0);
    }

    void update(Metrics metrics) {
        epochValue.setText(metrics.epoch() + " / " + metrics.totalEpochs());
        lossValue.setText(String.format("%.6f", metrics.loss()));
        trainAccuracyValue.setText(percent(metrics.trainAccuracy()));
        validationAccuracyValue.setText(percent(metrics.valAccuracy()));
        timeValue.setText(String.format("%.2f s", metrics.elapsedMillis() / MILLIS_PER_SECOND));
        progress.setProgress((double) metrics.epoch() / Math.max(1, metrics.totalEpochs()));
    }

    /** Validation accuracy is NaN when the dataset was trained without a held-out split. */
    private static String percent(double fraction) {
        return Double.isNaN(fraction) ? EMPTY : String.format("%.1f%%", fraction * 100);
    }

    private void addPair(int row, String caption, Label valueLabel) {
        Label key = new Label(caption);
        key.setStyle(Theme.HINT_LABEL_STYLE);
        add(key, 0, row);
        add(valueLabel, 1, row);
    }

    private static Label value() {
        Label label = new Label(EMPTY);
        label.setStyle(Theme.MONO_LABEL_STYLE);
        return label;
    }
}
