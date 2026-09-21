package com.neurojava.viz;

import com.neurojava.core.Matrix;
import com.neurojava.data.Split;
import com.neurojava.nn.NeuralNetwork;
import com.neurojava.training.Metrics;
import com.neurojava.training.TrainingConfig;

import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.SplitPane;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import javafx.util.Duration;

import java.util.List;
import java.util.Random;

/** The dashboard: configure a network on the left, watch it learn in the middle. */
public final class NeuroJavaApp extends Application {

    private static final double WINDOW_WIDTH = 1280;
    private static final double WINDOW_HEIGHT = 800;
    private static final double ANIMATION_STEP_MILLIS = 420;

    private final ControlPanel controlPanel = new ControlPanel();
    private final NetworkView networkView = new NetworkView();
    private final LossChart lossChart = new LossChart();
    private final DecisionBoundaryView boundaryView = new DecisionBoundaryView();
    private final MetricsPanel metricsPanel = new MetricsPanel();
    private final PredictPanel predictPanel = new PredictPanel(this::setStatus);

    private final Label statusLabel = new Label("Ready. Configure a network and press Initialize.");
    private final Label architectureLabel = new Label();

    private final Button initializeButton = new Button("Initialize");
    private final Button trainButton = new Button("Train");
    private final Button pauseButton = new Button("Pause");
    private final Button stopButton = new Button("Stop");
    private final Button resetButton = new Button("Reset");
    private final Button forwardButton = new Button("Run Forward Pass");
    private final Button backwardButton = new Button("Show Backward Pass");

    private NeuralNetwork network;
    private TrainingSpec spec;
    private Split split;
    private TrainingSession session;
    private Timeline animation;
    private boolean trained;

    @Override
    public void start(Stage stage) {
        wireButtons();

        Scene scene = new Scene(buildLayout(), WINDOW_WIDTH, WINDOW_HEIGHT);
        scene.getStylesheets().add(Theme.stylesheet());
        stage.setTitle("NeuroJava — neural network from scratch");
        stage.setScene(scene);
        stage.setMinWidth(1040);
        stage.setMinHeight(680);
        stage.show();

        updateButtons();
        networkView.clearNetwork();
        boundaryView.clearBoundary();
    }

    private BorderPane buildLayout() {
        BorderPane root = new BorderPane();
        root.setStyle("-fx-background-color: #1b1f27;");
        root.setTop(header());
        root.setLeft(scrollable(controlPanel));
        root.setCenter(centre());
        root.setRight(sidebar());
        root.setBottom(actionBar());
        return root;
    }

    private HBox header() {
        Label title = new Label("NEUROJAVA");
        title.setStyle(Theme.TITLE_LABEL_STYLE);
        architectureLabel.setStyle(Theme.MONO_LABEL_STYLE);
        statusLabel.setStyle(Theme.HINT_LABEL_STYLE);

        HBox box = new HBox(16, title, architectureLabel, spacer(), statusLabel);
        box.setPadding(new Insets(10, 14, 10, 14));
        box.setStyle(Theme.PANEL_STYLE);
        return box;
    }

    private SplitPane centre() {
        SplitPane lower = new SplitPane(lossChart, wrap(boundaryView));
        lower.setOrientation(Orientation.HORIZONTAL);
        lower.setDividerPositions(0.58);

        SplitPane centre = new SplitPane(wrap(networkView), lower);
        centre.setOrientation(Orientation.VERTICAL);
        centre.setDividerPositions(0.58);
        return centre;
    }

    private VBox sidebar() {
        VBox box = new VBox(10, metricsPanel, predictPanel);
        box.setPadding(new Insets(10));
        box.setPrefWidth(290);
        box.setStyle("-fx-background-color: #1b1f27;");
        return box;
    }

    private HBox actionBar() {
        HBox box = new HBox(8, initializeButton, trainButton, pauseButton, stopButton, resetButton,
                spacer(), forwardButton, backwardButton);
        box.setPadding(new Insets(10, 14, 10, 14));
        box.setStyle(Theme.PANEL_STYLE);
        return box;
    }

    private void wireButtons() {
        initializeButton.setOnAction(e -> initialize());
        trainButton.setOnAction(e -> train());
        pauseButton.setOnAction(e -> togglePause());
        stopButton.setOnAction(e -> stopTraining());
        resetButton.setOnAction(e -> reset());
        forwardButton.setOnAction(e -> animatePass(true));
        backwardButton.setOnAction(e -> animatePass(false));
    }

    /** Builds a fresh network from the form. Nothing else is enabled until this succeeds. */
    private void initialize() {
        controlPanel.read().ifPresentOrElse(parsed -> {
            try {
                spec = parsed;
                network = NetworkFactory.build(parsed);
                split = Split.of(parsed.dataset(), parsed.trainFraction(), new Random(parsed.seed()));
                trained = false;

                lossChart.clear();
                metricsPanel.reset();
                predictPanel.setContext(null, null);
                boundaryView.clearBoundary();
                architectureLabel.setText(parsed.dataset().name() + "   "
                        + NetworkFactory.describe(network.architecture()));
                renderNetwork(null);
                setStatus("Initialised with random weights — "
                        + split.trainInputs().rows() + " training rows"
                        + (split.hasValidation() ? ", " + split.valInputs().rows() + " validation rows" : ""));
            } catch (IllegalArgumentException | IllegalStateException ex) {
                network = null;
                setError(ex.getMessage());
            }
            updateButtons();
        }, () -> setError("Fix the highlighted settings, then press Initialize again."));
    }

    private void train() {
        if (network == null || spec == null || split == null) {
            return;
        }
        TrainingConfig config = new TrainingConfig(
                spec.epochs(), spec.learningRate(), spec.batchSize(), spec.seed());
        session = new TrainingSession(network, config, split,
                this::onEpochs, this::setError, this::onTrainingFinished);
        controlPanel.setFormDisabled(true);
        setStatus("Training…");
        updateButtons();
        session.start();
    }

    /**
     * One batch of epochs. Every epoch goes onto the chart so the curve keeps its real shape,
     * but the labels and the network are repainted once per batch rather than once per epoch.
     */
    private void onEpochs(List<Metrics> batch) {
        for (Metrics metrics : batch) {
            lossChart.add(metrics, metrics.epoch() == metrics.totalEpochs());
        }
        metricsPanel.update(batch.get(batch.size() - 1));
        renderNetwork(null);
    }

    private void onTrainingFinished() {
        trained = true;
        controlPanel.setFormDisabled(false);

        if (session != null && session.hasDiverged()) {
            setError("Training diverged (loss became NaN or infinite) — try a lower learning rate.");
        } else {
            setStatus("Training finished.");
        }
        predictPanel.setContext(network, spec.dataset());
        renderNetwork(firstSample());
        // Always hand the view the network: it explains for itself why a 4-feature dataset
        // has no 2-D boundary, which is more useful than leaving the initial prompt up.
        boundaryView.render(network, spec.dataset());
        session = null;
        updateButtons();
    }

    private void togglePause() {
        if (session == null) {
            return;
        }
        if (session.isPaused()) {
            session.resume();
            setStatus("Training…");
        } else {
            session.pause();
            setStatus("Paused.");
        }
        updateButtons();
    }

    private void stopTraining() {
        if (session != null) {
            session.stop();
            setStatus("Stopping after the current epoch…");
        }
    }

    private void reset() {
        stopAnimation();
        network = null;
        spec = null;
        split = null;
        trained = false;
        lossChart.clear();
        metricsPanel.reset();
        predictPanel.setContext(null, null);
        networkView.clearNetwork();
        boundaryView.clearBoundary();
        architectureLabel.setText("");
        controlPanel.setFormDisabled(false);
        setStatus("Reset. Configure a network and press Initialize.");
        updateButtons();
    }

    /**
     * Steps a highlight through the layers. Only the highlight is animated — animating every
     * individual multiply-add would make the screen unreadable without teaching anything.
     */
    private void animatePass(boolean forward) {
        if (network == null || session != null) {
            return;
        }
        stopAnimation();
        Matrix sample = firstSample();
        if (sample == null) {
            return;
        }
        network.forward(sample);

        int columns = network.architecture().length;
        animation = new Timeline();
        for (int step = 0; step < columns; step++) {
            int column = forward ? step : columns - 1 - step;
            String caption = forward
                    ? "Forward pass — " + (column == 0 ? "input" : "layer " + column)
                    : "Backward pass — gradients dL/dW" + Math.max(1, column);
            animation.getKeyFrames().add(new KeyFrame(
                    Duration.millis(ANIMATION_STEP_MILLIS * step),
                    e -> networkView.render(network, sample, column, caption)));
        }
        animation.getKeyFrames().add(new KeyFrame(
                Duration.millis(ANIMATION_STEP_MILLIS * columns),
                e -> networkView.render(network, sample, -1,
                        forward ? "Forward pass complete" : "Weights updated by gradient descent")));
        animation.play();
        setStatus(forward ? "Running a forward pass…" : "Showing the backward pass…");
    }

    private void stopAnimation() {
        if (animation != null) {
            animation.stop();
            animation = null;
        }
    }

    private Matrix firstSample() {
        if (split == null || split.trainInputs().rows() == 0) {
            return null;
        }
        return split.trainInputs().row(0);
    }

    private void renderNetwork(Matrix sample) {
        networkView.render(network, sample, -1, "");
    }

    private void updateButtons() {
        boolean training = session != null && session.isRunning();
        boolean ready = network != null;

        initializeButton.setDisable(training);
        trainButton.setDisable(training || !ready);
        pauseButton.setDisable(!training);
        stopButton.setDisable(!training);
        resetButton.setDisable(training);
        forwardButton.setDisable(training || !ready);
        backwardButton.setDisable(training || !ready || !trained);
        pauseButton.setText(session != null && session.isPaused() ? "Resume" : "Pause");
    }

    private void setStatus(String message) {
        statusLabel.setStyle(Theme.HINT_LABEL_STYLE);
        statusLabel.setText(message);
    }

    private void setError(String message) {
        statusLabel.setStyle(Theme.ERROR_LABEL_STYLE);
        statusLabel.setText(message == null ? "Something went wrong." : message);
    }

    private static ScrollPane scrollable(Region content) {
        ScrollPane pane = new ScrollPane(content);
        pane.setFitToWidth(true);
        pane.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        pane.setStyle("-fx-background-color: #232833;");
        return pane;
    }

    private static StackPane wrap(Node canvas) {
        StackPane pane = new StackPane(canvas);
        pane.setStyle("-fx-background-color: #1b1f27;");
        return pane;
    }

    private static Region spacer() {
        Region region = new Region();
        HBox.setHgrow(region, Priority.ALWAYS);
        return region;
    }
}
