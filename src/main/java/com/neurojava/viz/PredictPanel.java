package com.neurojava.viz;

import com.neurojava.core.Matrix;
import com.neurojava.data.Dataset;
import com.neurojava.nn.NeuralNetwork;

import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;

import java.util.function.Consumer;

/** Classify a hand-entered sample with the trained network. */
final class PredictPanel extends VBox {

    private static final double BINARY_THRESHOLD = 0.5;

    private final TextField inputField = new TextField();
    private final Button predictButton = new Button("Predict");
    private final Label hint = new Label();
    private final Label verdict = new Label();
    private final Label detail = new Label();
    private final Consumer<String> statusSink;

    private NeuralNetwork network;
    private Dataset dataset;

    PredictPanel(Consumer<String> statusSink) {
        this.statusSink = statusSink;
        setSpacing(6);
        setPadding(new Insets(10));
        setStyle(Theme.PANEL_STYLE);

        Label title = new Label("Predict");
        title.setStyle(Theme.SECTION_LABEL_STYLE);
        hint.setStyle(Theme.HINT_LABEL_STYLE);
        hint.setWrapText(true);
        verdict.setStyle(Theme.MONO_LABEL_STYLE);
        detail.setStyle(Theme.HINT_LABEL_STYLE);
        detail.setWrapText(true);

        HBox row = new HBox(6, inputField, predictButton);
        HBox.setHgrow(inputField, Priority.ALWAYS);
        inputField.setPromptText("e.g. 0, 1");
        inputField.setOnAction(e -> predict());
        predictButton.setOnAction(e -> predict());

        getChildren().addAll(title, hint, row, verdict, detail);
        setContext(null, null);
    }

    void setContext(NeuralNetwork network, Dataset dataset) {
        this.network = network;
        this.dataset = dataset;
        boolean ready = network != null && dataset != null;
        inputField.setDisable(!ready);
        predictButton.setDisable(!ready);
        verdict.setText("");
        detail.setText("");
        hint.setText(ready
                ? "Enter " + dataset.inputSize() + " values (" + String.join(", ", dataset.featureNames()) + ")"
                : "Train a network first.");
        if (ready) {
            inputField.setText(example());
        }
    }

    /** A real row from the dataset, so the field is never empty when a demo starts. */
    private String example() {
        Matrix inputs = dataset.inputs();
        if (inputs.rows() == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int c = 0; c < dataset.inputSize(); c++) {
            if (c > 0) {
                sb.append(", ");
            }
            // XOR is stored raw, so its own rows are the natural example; Iris is standardised,
            // so show the column mean in raw units instead of a z-score.
            sb.append(dataset.isBinary()
                    ? trim(inputs.get(0, c))
                    : trim(rawMean(c)));
        }
        return sb.toString();
    }

    private double rawMean(int column) {
        return dataset instanceof com.neurojava.data.IrisDataset iris
                ? iris.featureMeans()[column]
                : dataset.inputs().get(0, column);
    }

    private static String trim(double value) {
        return String.format("%.2f", value).replaceAll("0+$", "").replaceAll("\\.$", "");
    }

    private void predict() {
        if (network == null || dataset == null) {
            return;
        }
        String[] parts = inputField.getText().trim().split("[,\\s]+");
        if (inputField.getText().trim().isEmpty() || parts.length != dataset.inputSize()) {
            showError("Enter exactly " + dataset.inputSize() + " comma-separated values");
            return;
        }
        double[] raw = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                raw[i] = Double.parseDouble(parts[i]);
            } catch (NumberFormatException e) {
                showError("\"" + parts[i] + "\" is not a number");
                return;
            }
        }

        try {
            // The same scaling the network was trained on, or the values mean nothing to it.
            Matrix scaled = dataset.standardize(Matrix.rowVector(raw));
            Matrix output = network.predict(scaled);
            showResult(output);
        } catch (IllegalArgumentException | IllegalStateException e) {
            showError(e.getMessage());
        }
    }

    private void showResult(Matrix output) {
        verdict.setStyle(Theme.MONO_LABEL_STYLE);
        String[] classNames = dataset.classNames();
        if (output.cols() == 1) {
            double p = output.get(0, 0);
            int predicted = p >= BINARY_THRESHOLD ? 1 : 0;
            double confidence = predicted == 1 ? p : 1 - p;
            verdict.setText(String.format("Class %s  (%.1f%% confident)",
                    classNames.length > predicted ? classNames[predicted] : String.valueOf(predicted),
                    confidence * 100));
            detail.setText(String.format("Raw network output: %.6f", p));
            statusSink.accept("Predicted " + verdict.getText());
            return;
        }
        int winner = output.argMaxInRow(0);
        verdict.setText(String.format("%s  (%.1f%%)",
                classNames.length > winner ? classNames[winner] : "Class " + winner,
                output.get(0, winner) * 100));
        StringBuilder sb = new StringBuilder();
        for (int c = 0; c < output.cols(); c++) {
            if (c > 0) {
                sb.append("   ");
            }
            sb.append(String.format("%s %.1f%%",
                    classNames.length > c ? classNames[c] : "C" + c, output.get(0, c) * 100));
        }
        detail.setText(sb.toString());
        statusSink.accept("Predicted " + verdict.getText());
    }

    private void showError(String message) {
        verdict.setStyle(Theme.ERROR_LABEL_STYLE);
        verdict.setText(message);
        detail.setText("");
    }
}
