package com.neurojava.viz;

import com.neurojava.data.Split;
import com.neurojava.nn.NeuralNetwork;
import com.neurojava.training.Metrics;
import com.neurojava.training.Trainer;
import com.neurojava.training.TrainingConfig;

import javafx.application.Platform;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.Consumer;

/**
 * Owns the background training thread and the JavaFX thread hand-off.
 *
 * <p>{@link Trainer#train} blocks, so it runs on a daemon thread and every callback is marshalled
 * onto the FX thread. Pushing one {@code runLater} per epoch would swamp the FX queue, so epochs
 * are queued and delivered in batches about thirty times a second.
 *
 * <p>The throttle deliberately batches rather than DROPS: XOR trains in tens of milliseconds, so
 * dropping would leave the loss curve with two points and a straight line between them. Every
 * epoch still reaches the chart — only the number of FX hand-offs is limited.
 */
final class TrainingSession {

    private static final long UI_INTERVAL_MILLIS = 33;

    private final Trainer trainer;
    private final Thread thread;
    private volatile Metrics latest;

    TrainingSession(NeuralNetwork network,
                    TrainingConfig config,
                    Split split,
                    Consumer<List<Metrics>> onEpochs,
                    Consumer<String> onError,
                    Runnable onFinished) {
        this.trainer = new Trainer(network, config);
        this.thread = new Thread(() -> {
            long[] lastPush = {0};
            Queue<Metrics> pending = new ConcurrentLinkedQueue<>();
            try {
                trainer.train(split.trainInputs(), split.trainTargets(),
                        split.valInputs(), split.valTargets(),
                        metrics -> {
                            latest = metrics;
                            pending.add(metrics);
                            long now = System.currentTimeMillis();
                            if (now - lastPush[0] >= UI_INTERVAL_MILLIS) {
                                lastPush[0] = now;
                                List<Metrics> batch = drain(pending);
                                if (!batch.isEmpty()) {
                                    Platform.runLater(() -> onEpochs.accept(batch));
                                }
                            }
                        });
            } catch (RuntimeException e) {
                String message = e.getMessage() == null ? e.toString() : e.getMessage();
                Platform.runLater(() -> onError.accept(message));
            } finally {
                // Whatever the throttle had not yet handed over, including the final epoch.
                List<Metrics> tail = drain(pending);
                Platform.runLater(() -> {
                    if (!tail.isEmpty()) {
                        onEpochs.accept(tail);
                    }
                    onFinished.run();
                });
            }
        }, "neurojava-training");
        this.thread.setDaemon(true);
    }

    private static List<Metrics> drain(Queue<Metrics> queue) {
        List<Metrics> batch = new ArrayList<>();
        for (Metrics metrics = queue.poll(); metrics != null; metrics = queue.poll()) {
            batch.add(metrics);
        }
        return batch;
    }

    void start() {
        thread.start();
    }

    void pause() {
        trainer.pause();
    }

    void resume() {
        trainer.resume();
    }

    void stop() {
        trainer.stop();
    }

    boolean isPaused() {
        return trainer.isPaused();
    }

    boolean isRunning() {
        return trainer.isRunning();
    }

    boolean hasDiverged() {
        return trainer.hasDiverged();
    }
}
