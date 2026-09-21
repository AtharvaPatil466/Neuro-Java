package com.neurojava.viz;

import com.neurojava.core.Matrix;
import com.neurojava.nn.DenseLayer;
import com.neurojava.nn.Layer;
import com.neurojava.nn.NeuralNetwork;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;

/**
 * The network itself: neurons as circles, weights as lines.
 *
 * <p>Weight magnitude is line thickness and weight sign is line STYLE — solid for positive,
 * dashed for negative. Sign is deliberately never carried by colour alone, so the picture still
 * reads on a projector or to a colour-blind viewer.
 */
final class NetworkView extends ResizableCanvas {

    private static final double PADDING_X = 72;
    private static final double PADDING_TOP = 54;
    private static final double PADDING_BOTTOM = 46;
    private static final double MAX_RADIUS = 20;
    private static final double MIN_RADIUS = 3.5;
    private static final double MIN_STROKE = 0.4;
    private static final double MAX_STROKE = 5.0;
    /** Above this many neurons in a column the per-neuron numbers would overlap, so they are dropped. */
    private static final int MAX_LABELLED_NEURONS = 12;
    private static final double EPSILON = 1e-12;

    private NeuralNetwork network;
    private Matrix sample;
    private int highlightColumn = -1;
    private String caption = "";

    NetworkView() {
        super(720, 420);
    }

    /**
     * @param sample the (1 x inputs) row currently pushed through the net, or null
     * @param highlightColumn column to ring for the forward/backward animation, or -1
     */
    void render(NeuralNetwork network, Matrix sample, int highlightColumn, String caption) {
        this.network = network;
        this.sample = sample;
        this.highlightColumn = highlightColumn;
        this.caption = caption == null ? "" : caption;
        redraw();
    }

    void clearNetwork() {
        render(null, null, -1, "");
    }

    @Override
    protected void redraw() {
        GraphicsContext gc = getGraphicsContext2D();
        double w = getWidth();
        double h = getHeight();
        gc.setFill(Theme.BACKGROUND);
        gc.fillRect(0, 0, w, h);

        if (network == null) {
            drawPlaceholder(gc, w, h);
            return;
        }

        int[] architecture = network.architecture();
        double[] columnX = columnPositions(architecture.length, w);
        double radius = neuronRadius(architecture, h);

        drawConnections(gc, architecture, columnX, h, radius);
        drawNeurons(gc, architecture, columnX, h, radius);
        drawColumnCaptions(gc, architecture, columnX);
        drawLegend(gc, h);
        drawCaption(gc, w);
    }

    private void drawPlaceholder(GraphicsContext gc, double w, double h) {
        gc.setFill(Theme.MUTED);
        gc.setFont(Font.font(13));
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText("Configure a network and press Initialize", w / 2, h / 2);
        gc.setTextAlign(TextAlignment.LEFT);
    }

    private static double[] columnPositions(int columns, double width) {
        double[] xs = new double[columns];
        double usable = Math.max(1, width - 2 * PADDING_X);
        for (int i = 0; i < columns; i++) {
            xs[i] = columns == 1 ? width / 2 : PADDING_X + usable * i / (columns - 1);
        }
        return xs;
    }

    private static double neuronRadius(int[] architecture, double height) {
        int widest = 1;
        for (int count : architecture) {
            widest = Math.max(widest, count);
        }
        double usable = Math.max(1, height - PADDING_TOP - PADDING_BOTTOM);
        return Math.max(MIN_RADIUS, Math.min(MAX_RADIUS, usable / (2.6 * widest)));
    }

    private double neuronY(int index, int count, double height) {
        double usable = height - PADDING_TOP - PADDING_BOTTOM;
        if (count == 1) {
            return PADDING_TOP + usable / 2;
        }
        return PADDING_TOP + usable * index / (count - 1);
    }

    private void drawConnections(GraphicsContext gc, int[] architecture, double[] columnX,
                                 double height, double radius) {
        double largest = largestAbsoluteWeight();
        for (int layerIndex = 0; layerIndex < network.layers().size(); layerIndex++) {
            Layer layer = network.layers().get(layerIndex);
            if (!(layer instanceof DenseLayer dense)) {
                continue;
            }
            Matrix weights = dense.weights();
            int from = architecture[layerIndex];
            int to = architecture[layerIndex + 1];
            for (int i = 0; i < from; i++) {
                double y1 = neuronY(i, from, height);
                for (int j = 0; j < to; j++) {
                    double y2 = neuronY(j, to, height);
                    double weight = weights.get(i, j);
                    double normalised = Math.abs(weight) / largest;
                    gc.setLineWidth(MIN_STROKE + normalised * (MAX_STROKE - MIN_STROKE));
                    gc.setStroke((weight >= 0 ? Theme.POSITIVE : Theme.NEGATIVE)
                            .deriveColor(0, 1, 1, 0.25 + 0.6 * normalised));
                    if (weight < 0) {
                        gc.setLineDashes(6, 5);
                    } else {
                        gc.setLineDashes();
                    }
                    gc.strokeLine(columnX[layerIndex] + radius, y1, columnX[layerIndex + 1] - radius, y2);
                }
            }
        }
        gc.setLineDashes();
    }

    /** Normalising by the largest magnitude keeps thickness meaningful as the weights grow. */
    private double largestAbsoluteWeight() {
        double largest = EPSILON;
        for (Layer layer : network.layers()) {
            if (!(layer instanceof DenseLayer dense)) {
                continue;
            }
            Matrix weights = dense.weights();
            for (int r = 0; r < weights.rows(); r++) {
                for (int c = 0; c < weights.cols(); c++) {
                    largest = Math.max(largest, Math.abs(weights.get(r, c)));
                }
            }
        }
        return largest;
    }

    private void drawNeurons(GraphicsContext gc, int[] architecture, double[] columnX,
                            double height, double radius) {
        gc.setFont(Font.font(10.5));
        for (int column = 0; column < architecture.length; column++) {
            int count = architecture[column];
            double[] activations = activationsFor(column, count);
            boolean labelled = count <= MAX_LABELLED_NEURONS;
            for (int i = 0; i < count; i++) {
                double x = columnX[column];
                double y = neuronY(i, count, height);
                double intensity = intensity(activations, i);

                gc.setFill(Theme.NEURON_BASE.interpolate(Theme.ACCENT, intensity));
                gc.fillOval(x - radius, y - radius, radius * 2, radius * 2);
                gc.setLineDashes();
                gc.setLineWidth(column == highlightColumn ? 3 : 1.2);
                gc.setStroke(column == highlightColumn ? Theme.HIGHLIGHT : Theme.GRID);
                gc.strokeOval(x - radius, y - radius, radius * 2, radius * 2);

                if (labelled && activations != null) {
                    gc.setFill(Theme.TEXT);
                    gc.setTextAlign(TextAlignment.LEFT);
                    gc.fillText(String.format("%.2f", activations[i]), x + radius + 5, y + 3.5);
                }
            }
        }
        gc.setTextAlign(TextAlignment.LEFT);
    }

    private static double intensity(double[] activations, int index) {
        if (activations == null) {
            return 0;
        }
        double largest = EPSILON;
        for (double value : activations) {
            largest = Math.max(largest, Math.abs(value));
        }
        return Math.min(1, Math.abs(activations[index]) / largest);
    }

    /** Column 0 is the input sample; every later column is the matching layer's cached output. */
    private double[] activationsFor(int column, int count) {
        Matrix source = column == 0 ? sample : cachedActivation(column - 1);
        if (source == null || source.rows() == 0 || source.cols() != count) {
            return null;
        }
        double[] values = new double[count];
        for (int i = 0; i < count; i++) {
            values[i] = source.get(0, i);
        }
        return values;
    }

    private Matrix cachedActivation(int layerIndex) {
        Layer layer = network.layers().get(layerIndex);
        return layer instanceof DenseLayer dense ? dense.lastActivation() : null;
    }

    private void drawColumnCaptions(GraphicsContext gc, int[] architecture, double[] columnX) {
        gc.setFont(Font.font(11.5));
        gc.setTextAlign(TextAlignment.CENTER);
        for (int column = 0; column < architecture.length; column++) {
            gc.setFill(column == highlightColumn ? Theme.HIGHLIGHT : Theme.MUTED);
            gc.fillText(columnName(column, architecture.length), columnX[column], 24);
            gc.setFill(Theme.MUTED);
            gc.fillText(architecture[column] + activationSuffix(column), columnX[column], 39);
        }
        gc.setTextAlign(TextAlignment.LEFT);
    }

    private static String columnName(int column, int columns) {
        if (column == 0) {
            return "Input";
        }
        return column == columns - 1 ? "Output" : "Hidden " + column;
    }

    private String activationSuffix(int column) {
        if (column == 0) {
            return " neurons";
        }
        Layer layer = network.layers().get(column - 1);
        return layer instanceof DenseLayer dense ? " · " + dense.activation().name() : "";
    }

    private void drawLegend(GraphicsContext gc, double height) {
        double y = height - 20;
        gc.setFont(Font.font(10.5));
        gc.setLineWidth(2);

        gc.setLineDashes();
        gc.setStroke(Theme.POSITIVE);
        gc.strokeLine(PADDING_X - 50, y, PADDING_X - 20, y);
        gc.setFill(Theme.MUTED);
        gc.fillText("positive weight", PADDING_X - 14, y + 4);

        gc.setLineDashes(6, 5);
        gc.setStroke(Theme.NEGATIVE);
        gc.strokeLine(PADDING_X + 70, y, PADDING_X + 100, y);
        gc.setLineDashes();
        gc.fillText("negative weight   ·   thickness = |weight|   ·   fill = activation",
                PADDING_X + 106, y + 4);
    }

    private void drawCaption(GraphicsContext gc, double width) {
        if (caption.isEmpty()) {
            return;
        }
        gc.setFont(Font.font(12.5));
        gc.setFill(Theme.HIGHLIGHT);
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(caption, width / 2, 14);
        gc.setTextAlign(TextAlignment.LEFT);
    }
}
