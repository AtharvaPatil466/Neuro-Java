package com.neurojava.viz;

import com.neurojava.data.Dataset;
import com.neurojava.data.IrisDataset;
import com.neurojava.data.XorDataset;

import javafx.geometry.Insets;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * The configuration form. Every field validates on read and reports problems as an inline label
 * next to the field — never a modal dialog, never an exception reaching the user.
 */
final class ControlPanel extends VBox {

    private static final String XOR = "XOR";
    private static final String IRIS = "Iris";
    private static final int MAX_HIDDEN_LAYERS = 4;
    private static final int MAX_NEURONS_PER_LAYER = 64;
    private static final int MAX_EPOCHS = 1_000_000;
    private static final double MAX_LEARNING_RATE = 10.0;
    /** XOR has four samples in total, so holding any back would leave nothing to learn from. */
    private static final int MIN_ROWS_FOR_VALIDATION = 10;
    private static final double TRAIN_FRACTION = 0.8;

    private final ChoiceBox<String> datasetChoice = new ChoiceBox<>();
    private final ChoiceBox<String> activationChoice = new ChoiceBox<>();
    private final TextField hiddenField = new TextField();
    private final TextField learningRateField = new TextField();
    private final TextField epochsField = new TextField();
    private final TextField batchSizeField = new TextField();
    private final TextField seedField = new TextField();
    private final CheckBox validationSplitBox = new CheckBox("Hold out 20% for validation");

    private final Label hiddenError = error();
    private final Label learningRateError = error();
    private final Label epochsError = error();
    private final Label batchSizeError = error();
    private final Label seedError = error();
    private final Label architecturePreview = new Label();

    private Dataset irisCache;

    ControlPanel() {
        setSpacing(8);
        setPadding(new Insets(12));
        setStyle(Theme.PANEL_STYLE);
        setPrefWidth(268);

        datasetChoice.getItems().addAll(XOR, IRIS);
        datasetChoice.setValue(XOR);
        datasetChoice.setMaxWidth(Double.MAX_VALUE);
        datasetChoice.valueProperty().addListener((obs, old, now) -> applyDatasetDefaults());

        activationChoice.getItems().addAll("Sigmoid", "ReLU");
        activationChoice.setValue("Sigmoid");
        activationChoice.setMaxWidth(Double.MAX_VALUE);

        architecturePreview.setStyle(Theme.MONO_LABEL_STYLE);

        getChildren().addAll(
                section("Dataset"), datasetChoice,
                section("Hidden layers"), hiddenField,
                hint("Comma separated, e.g. 4 or 8,6. Empty means no hidden layer."), hiddenError,
                section("Hidden activation"), activationChoice,
                section("Learning rate"), learningRateField, learningRateError,
                section("Epochs"), epochsField, epochsError,
                section("Batch size"), batchSizeField, batchSizeError,
                section("Random seed"), seedField, seedError,
                validationSplitBox,
                section("Architecture"), architecturePreview,
                hint("The output layer is chosen by the dataset: sigmoid for binary, softmax for multiclass."));

        hiddenField.textProperty().addListener((obs, old, now) -> refreshPreview());
        applyDatasetDefaults();
    }

    /** Sensible starting points, so the demo works on the first press of Train. */
    private void applyDatasetDefaults() {
        boolean xor = XOR.equals(datasetChoice.getValue());
        hiddenField.setText(xor ? "4" : "8,6");
        activationChoice.setValue(xor ? "Sigmoid" : "ReLU");
        // Measured over 20 seeds: sigmoid at 0.5 for 2000 epochs solves XOR every time, while the
        // textbook 0.01 never converges and ReLU stalls on 4 of 20 seeds with dead units.
        learningRateField.setText(xor ? "0.5" : "0.05");
        epochsField.setText(xor ? "2000" : "500");
        batchSizeField.setText(xor ? "4" : "16");
        seedField.setText("42");

        boolean canSplit = dataset().inputs().rows() >= MIN_ROWS_FOR_VALIDATION;
        validationSplitBox.setDisable(!canSplit);
        validationSplitBox.setSelected(canSplit);
        refreshPreview();
    }

    /** Datasets are cached: re-parsing the Iris CSV on every keystroke would be wasteful. */
    Dataset dataset() {
        if (XOR.equals(datasetChoice.getValue())) {
            return new XorDataset();
        }
        if (irisCache == null) {
            irisCache = new IrisDataset();
        }
        return irisCache;
    }

    /** Returns the parsed configuration, or empty after painting the problems onto the form. */
    Optional<TrainingSpec> read() {
        clearErrors();
        List<String> failures = new ArrayList<>();

        Dataset dataset = dataset();
        int[] hidden = parseHidden(failures);
        double learningRate = parsePositiveDouble(learningRateField, learningRateError,
                "Learning rate", MAX_LEARNING_RATE, failures);
        int epochs = parsePositiveInt(epochsField, epochsError, "Epochs", MAX_EPOCHS, failures);
        int batchSize = parsePositiveInt(batchSizeField, batchSizeError, "Batch size",
                Math.max(1, dataset.inputs().rows()), failures);
        long seed = parseSeed(failures);

        if (!failures.isEmpty()) {
            return Optional.empty();
        }
        double trainFraction = validationSplitBox.isSelected() && !validationSplitBox.isDisabled()
                ? TRAIN_FRACTION
                : 1.0;
        return Optional.of(new TrainingSpec(dataset, hidden, activationChoice.getValue(),
                learningRate, epochs, batchSize, seed, trainFraction));
    }

    private int[] parseHidden(List<String> failures) {
        String raw = hiddenField.getText().trim();
        if (raw.isEmpty()) {
            return new int[0];
        }
        String[] parts = raw.split(",");
        if (parts.length > MAX_HIDDEN_LAYERS) {
            fail(hiddenError, "At most " + MAX_HIDDEN_LAYERS + " hidden layers", failures);
            return new int[0];
        }
        int[] sizes = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                sizes[i] = Integer.parseInt(parts[i].trim());
            } catch (NumberFormatException e) {
                fail(hiddenError, "\"" + parts[i].trim() + "\" is not a whole number", failures);
                return new int[0];
            }
            if (sizes[i] < 1 || sizes[i] > MAX_NEURONS_PER_LAYER) {
                fail(hiddenError, "Each layer needs 1 to " + MAX_NEURONS_PER_LAYER + " neurons", failures);
                return new int[0];
            }
        }
        return sizes;
    }

    private double parsePositiveDouble(TextField field, Label errorLabel, String name,
                                       double max, List<String> failures) {
        try {
            double value = Double.parseDouble(field.getText().trim());
            if (!(value > 0) || value > max || !Double.isFinite(value)) {
                fail(errorLabel, name + " must be between 0 and " + max, failures);
                return 0;
            }
            return value;
        } catch (NumberFormatException e) {
            fail(errorLabel, name + " must be a number", failures);
            return 0;
        }
    }

    private int parsePositiveInt(TextField field, Label errorLabel, String name,
                                 int max, List<String> failures) {
        try {
            int value = Integer.parseInt(field.getText().trim());
            if (value < 1 || value > max) {
                fail(errorLabel, name + " must be between 1 and " + max, failures);
                return 1;
            }
            return value;
        } catch (NumberFormatException e) {
            fail(errorLabel, name + " must be a whole number", failures);
            return 1;
        }
    }

    private long parseSeed(List<String> failures) {
        try {
            return Long.parseLong(seedField.getText().trim());
        } catch (NumberFormatException e) {
            fail(seedError, "Seed must be a whole number", failures);
            return 0;
        }
    }

    private void refreshPreview() {
        Dataset dataset = dataset();
        StringBuilder sb = new StringBuilder().append(dataset.inputSize());
        String raw = hiddenField.getText().trim();
        if (!raw.isEmpty()) {
            for (String part : raw.split(",")) {
                sb.append(" -> ").append(part.trim().isEmpty() ? "?" : part.trim());
            }
        }
        sb.append(" -> ").append(dataset.outputSize());
        architecturePreview.setText(sb.toString());
    }

    void setFormDisabled(boolean disabled) {
        datasetChoice.setDisable(disabled);
        activationChoice.setDisable(disabled);
        hiddenField.setDisable(disabled);
        learningRateField.setDisable(disabled);
        epochsField.setDisable(disabled);
        batchSizeField.setDisable(disabled);
        seedField.setDisable(disabled);
        validationSplitBox.setDisable(disabled
                || dataset().inputs().rows() < MIN_ROWS_FOR_VALIDATION);
    }

    private void clearErrors() {
        for (Label label : List.of(hiddenError, learningRateError, epochsError, batchSizeError, seedError)) {
            label.setText("");
            label.setManaged(false);
        }
    }

    private static void fail(Label label, String message, List<String> failures) {
        label.setText(message);
        label.setManaged(true);
        failures.add(message);
    }

    private static Label error() {
        Label label = new Label();
        label.setStyle(Theme.ERROR_LABEL_STYLE);
        label.setWrapText(true);
        label.setManaged(false);
        return label;
    }

    private static Label section(String text) {
        Label label = new Label(text);
        label.setStyle(Theme.SECTION_LABEL_STYLE);
        VBox.setMargin(label, new Insets(6, 0, 0, 0));
        return label;
    }

    private static Label hint(String text) {
        Label label = new Label(text);
        label.setStyle(Theme.HINT_LABEL_STYLE);
        label.setWrapText(true);
        return label;
    }
}
