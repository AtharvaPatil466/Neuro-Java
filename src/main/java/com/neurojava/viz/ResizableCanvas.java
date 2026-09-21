package com.neurojava.viz;

import javafx.scene.canvas.Canvas;

/** A {@link Canvas} that follows its parent's layout and repaints when the size changes. */
abstract class ResizableCanvas extends Canvas {

    private static final double MIN_SIDE = 80;

    ResizableCanvas(double width, double height) {
        super(width, height);
    }

    /** Repaints the whole canvas from whatever state the subclass last stored. */
    protected abstract void redraw();

    @Override
    public boolean isResizable() {
        return true;
    }

    @Override
    public void resize(double width, double height) {
        setWidth(Math.max(MIN_SIDE, width));
        setHeight(Math.max(MIN_SIDE, height));
        redraw();
    }

    @Override
    public double minWidth(double height) {
        return MIN_SIDE;
    }

    @Override
    public double minHeight(double width) {
        return MIN_SIDE;
    }

    @Override
    public double prefWidth(double height) {
        return getWidth();
    }

    @Override
    public double prefHeight(double width) {
        return getHeight();
    }

    @Override
    public double maxWidth(double height) {
        return Double.MAX_VALUE;
    }

    @Override
    public double maxHeight(double width) {
        return Double.MAX_VALUE;
    }
}
