package com.neurojava.viz;

import com.neurojava.core.Matrix;
import com.neurojava.data.Dataset;
import com.neurojava.nn.NeuralNetwork;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;

/**
 * What the network actually learned, for datasets with two input features.
 *
 * <p>Samples a grid across the input plane, classifies every point in ONE batched forward pass,
 * paints the regions, then overlays the real training points. For XOR this is the picture that
 * shows a linear model could never do the job.
 */
final class DecisionBoundaryView extends ResizableCanvas {

    private static final int GRID = 120;
    private static final int TWO_FEATURES = 2;
    private static final double MARGIN_FRACTION = 0.15;
    private static final double FALLBACK_RANGE = 1.0;
    private static final double POINT_RADIUS = 5.5;
    private static final double PLOT_PADDING = 34;

    private NeuralNetwork network;
    private Dataset dataset;

    DecisionBoundaryView() {
        super(420, 360);
    }

    void render(NeuralNetwork network, Dataset dataset) {
        this.network = network;
        this.dataset = dataset;
        redraw();
    }

    void clearBoundary() {
        render(null, null);
    }

    @Override
    protected void redraw() {
        GraphicsContext gc = getGraphicsContext2D();
        double w = getWidth();
        double h = getHeight();
        gc.setFill(Theme.BACKGROUND);
        gc.fillRect(0, 0, w, h);

        if (network == null || dataset == null) {
            message(gc, "Train a network to see its decision boundary", w, h);
            return;
        }
        if (dataset.inputSize() != TWO_FEATURES) {
            message(gc, "Decision boundary is only available for 2-D datasets"
                    + System.lineSeparator() + "(" + dataset.name() + " has "
                    + dataset.inputSize() + " input features)", w, h);
            return;
        }

        Matrix inputs = dataset.inputs();
        double[] xRange = range(inputs, 0);
        double[] yRange = range(inputs, 1);
        double plotW = Math.max(1, w - 2 * PLOT_PADDING);
        double plotH = Math.max(1, h - 2 * PLOT_PADDING);

        gc.drawImage(regionImage(xRange, yRange), PLOT_PADDING, PLOT_PADDING, plotW, plotH);
        drawAxes(gc, xRange, yRange, plotW, plotH);
        drawSamples(gc, inputs, xRange, yRange, plotW, plotH);
        drawLegend(gc, w);
    }

    /** One batched predict over the whole grid — far cheaper than one call per pixel. */
    private WritableImage regionImage(double[] xRange, double[] yRange) {
        double[][] points = new double[GRID * GRID][TWO_FEATURES];
        for (int py = 0; py < GRID; py++) {
            for (int px = 0; px < GRID; px++) {
                points[py * GRID + px][0] = xRange[0] + (px + 0.5) / GRID * (xRange[1] - xRange[0]);
                points[py * GRID + px][1] = yRange[1] - (py + 0.5) / GRID * (yRange[1] - yRange[0]);
            }
        }
        Matrix output = network.predict(new Matrix(points));

        WritableImage image = new WritableImage(GRID, GRID);
        PixelWriter writer = image.getPixelWriter();
        for (int py = 0; py < GRID; py++) {
            for (int px = 0; px < GRID; px++) {
                writer.setColor(px, py, regionColor(output, py * GRID + px));
            }
        }
        return image;
    }

    /**
     * Binary heads blend between the two class colours by probability; softmax heads take the
     * winning class and fade with its confidence. Either way, uncertainty reads as a washed-out
     * band along the boundary.
     */
    private Color regionColor(Matrix output, int row) {
        if (output.cols() == 1) {
            double p = output.get(row, 0);
            Color blended = Theme.classColor(0).interpolate(Theme.classColor(1), clamp(p));
            double confidence = Math.abs(p - 0.5) * 2;
            return Theme.BACKGROUND.interpolate(blended, 0.25 + 0.45 * clamp(confidence));
        }
        int winner = output.argMaxInRow(row);
        double confidence = clamp(output.get(row, winner));
        return Theme.BACKGROUND.interpolate(Theme.classColor(winner), 0.25 + 0.45 * confidence);
    }

    private void drawSamples(GraphicsContext gc, Matrix inputs, double[] xRange, double[] yRange,
                             double plotW, double plotH) {
        Matrix targets = dataset.targets();
        for (int row = 0; row < inputs.rows(); row++) {
            double x = PLOT_PADDING + (inputs.get(row, 0) - xRange[0]) / (xRange[1] - xRange[0]) * plotW;
            double y = PLOT_PADDING + (yRange[1] - inputs.get(row, 1)) / (yRange[1] - yRange[0]) * plotH;
            int label = targets.cols() == 1
                    ? (targets.get(row, 0) >= 0.5 ? 1 : 0)
                    : targets.argMaxInRow(row);

            gc.setFill(Theme.classColor(label));
            gc.fillOval(x - POINT_RADIUS, y - POINT_RADIUS, POINT_RADIUS * 2, POINT_RADIUS * 2);
            gc.setLineDashes();
            gc.setLineWidth(1.6);
            gc.setStroke(Color.WHITE);
            gc.strokeOval(x - POINT_RADIUS, y - POINT_RADIUS, POINT_RADIUS * 2, POINT_RADIUS * 2);
        }
    }

    private void drawAxes(GraphicsContext gc, double[] xRange, double[] yRange,
                          double plotW, double plotH) {
        gc.setStroke(Theme.GRID);
        gc.setLineDashes();
        gc.setLineWidth(1);
        gc.strokeRect(PLOT_PADDING, PLOT_PADDING, plotW, plotH);

        gc.setFill(Theme.MUTED);
        gc.setFont(Font.font(10));
        String[] names = dataset.featureNames();
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(names.length > 0 ? names[0] : "Feature 1",
                PLOT_PADDING + plotW / 2, PLOT_PADDING + plotH + 22);
        gc.setTextAlign(TextAlignment.LEFT);
        gc.fillText(String.format("%.1f", xRange[0]), PLOT_PADDING, PLOT_PADDING + plotH + 12);
        gc.fillText(String.format("%.1f", xRange[1]), PLOT_PADDING + plotW - 16, PLOT_PADDING + plotH + 12);
        gc.fillText(names.length > 1 ? names[1] : "Feature 2", PLOT_PADDING - 26, PLOT_PADDING - 12);
    }

    private void drawLegend(GraphicsContext gc, double width) {
        String[] classNames = dataset.classNames();
        gc.setFont(Font.font(10.5));
        double x = PLOT_PADDING;
        double y = 16;
        for (int i = 0; i < classNames.length; i++) {
            gc.setFill(Theme.classColor(i));
            gc.fillOval(x, y - 7, 9, 9);
            gc.setFill(Theme.MUTED);
            gc.fillText(classNames[i], x + 14, y + 1);
            x += 24 + classNames[i].length() * 6.4;
            if (x > width - 60) {
                break;
            }
        }
    }

    /** Min and max of one feature column, padded so points never sit on the frame. */
    private static double[] range(Matrix inputs, int column) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (int row = 0; row < inputs.rows(); row++) {
            min = Math.min(min, inputs.get(row, column));
            max = Math.max(max, inputs.get(row, column));
        }
        if (!Double.isFinite(min) || !Double.isFinite(max)) {
            return new double[]{-FALLBACK_RANGE, FALLBACK_RANGE};
        }
        double span = max - min;
        double margin = span < 1e-9 ? FALLBACK_RANGE : span * MARGIN_FRACTION;
        return new double[]{min - margin, max + margin};
    }

    private static double clamp(double value) {
        return Math.max(0, Math.min(1, value));
    }

    private void message(GraphicsContext gc, String text, double w, double h) {
        gc.setFill(Theme.MUTED);
        gc.setFont(Font.font(12));
        gc.setTextAlign(TextAlignment.CENTER);
        double y = h / 2;
        for (String line : text.split(System.lineSeparator())) {
            gc.fillText(line, w / 2, y);
            y += 17;
        }
        gc.setTextAlign(TextAlignment.LEFT);
    }
}
