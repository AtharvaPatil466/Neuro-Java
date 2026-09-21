package com.neurojava;

import com.neurojava.viz.NeuroJavaApp;

import javafx.application.Application;

/**
 * Launcher.
 *
 * <p>This class deliberately does NOT extend {@link Application}: when JavaFX sits on the plain
 * classpath (rather than the module path) a main class that extends {@code Application} makes the
 * launcher abort with "JavaFX runtime components are missing". Delegating to
 * {@link Application#launch(Class, String...)} from an ordinary class sidesteps that check.
 */
public final class Main {

    private Main() {
    }

    public static void main(String[] args) {
        Application.launch(NeuroJavaApp.class, args);
    }
}
