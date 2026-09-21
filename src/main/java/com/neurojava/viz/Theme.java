package com.neurojava.viz;

import javafx.scene.paint.Color;

import java.nio.charset.StandardCharsets;
import java.util.Base64;

/** Shared colours and the single inline stylesheet, so the canvases and the controls agree. */
final class Theme {

    static final Color BACKGROUND = Color.web("#1b1f27");
    static final Color PANEL = Color.web("#232833");
    static final Color GRID = Color.web("#2e3542");
    static final Color TEXT = Color.web("#dfe4ee");
    static final Color MUTED = Color.web("#8b93a7");
    static final Color ACCENT = Color.web("#4da3ff");
    static final Color NEURON_BASE = Color.web("#39414f");
    static final Color HIGHLIGHT = Color.web("#ffd166");

    /** Positive weights. Sign is also encoded by a solid stroke, never by colour alone. */
    static final Color POSITIVE = Color.web("#4da3ff");

    /** Negative weights. Sign is also encoded by a dashed stroke. */
    static final Color NEGATIVE = Color.web("#ff9d4d");

    static final Color ERROR = Color.web("#ff6b6b");

    /** Class colours for the decision boundary, cycled if a dataset ever has more classes. */
    private static final Color[] CLASS_COLORS = {
            Color.web("#4da3ff"),
            Color.web("#ff9d4d"),
            Color.web("#6ee7a8"),
            Color.web("#c792ea"),
            Color.web("#f06292"),
    };

    static final String ERROR_LABEL_STYLE = "-fx-text-fill: #ff6b6b; -fx-font-size: 11px;";
    static final String HINT_LABEL_STYLE = "-fx-text-fill: #8b93a7; -fx-font-size: 11px;";
    static final String SECTION_LABEL_STYLE =
            "-fx-text-fill: #dfe4ee; -fx-font-size: 13px; -fx-font-weight: bold;";
    static final String TITLE_LABEL_STYLE =
            "-fx-text-fill: #ffffff; -fx-font-size: 17px; -fx-font-weight: bold;";
    static final String MONO_LABEL_STYLE = "-fx-font-family: 'Monospaced'; -fx-text-fill: #dfe4ee;";
    static final String PANEL_STYLE = "-fx-background-color: #232833;";

    private static final String CSS = String.join("\n",
            ".root { -fx-base: #262b36; -fx-background: #1b1f27; -fx-accent: #4da3ff;",
            "        -fx-focus-color: #4da3ff; -fx-faint-focus-color: #4da3ff22; -fx-font-size: 12.5px; }",
            ".label { -fx-text-fill: #dfe4ee; }",
            ".button { -fx-background-radius: 4; }",
            ".chart { -fx-background-color: #232833; -fx-padding: 6; }",
            ".chart-plot-background { -fx-background-color: #161a21; }",
            ".chart-vertical-grid-lines { -fx-stroke: #2e3542; }",
            ".chart-horizontal-grid-lines { -fx-stroke: #2e3542; }",
            ".chart-series-line { -fx-stroke-width: 2px; }",
            ".default-color0.chart-series-line { -fx-stroke: #4da3ff; }",
            ".default-color1.chart-series-line { -fx-stroke: #6ee7a8; }",
            ".default-color0.chart-legend-item-symbol { -fx-background-color: #4da3ff; }",
            ".default-color1.chart-legend-item-symbol { -fx-background-color: #6ee7a8; }",
            ".chart-legend { -fx-background-color: transparent; }",
            ".chart-legend-item { -fx-text-fill: #8b93a7; }",
            ".axis { -fx-tick-label-fill: #8b93a7; }",
            ".axis-label { -fx-text-fill: #8b93a7; }",
            ".progress-bar > .bar { -fx-background-color: #4da3ff; }",
            ".tab-header-background { -fx-background-color: #1b1f27; }",
            ".split-pane { -fx-background-color: #1b1f27; }");

    private Theme() {
    }

    static Color classColor(int index) {
        int safe = Math.floorMod(index, CLASS_COLORS.length);
        return CLASS_COLORS[safe];
    }

    /** The stylesheet as a data URI, so the project needs no .css resource on the classpath. */
    static String stylesheet() {
        return "data:text/css;base64,"
                + Base64.getEncoder().encodeToString(CSS.getBytes(StandardCharsets.UTF_8));
    }
}
