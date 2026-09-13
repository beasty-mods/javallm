package com.chatllm.core;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Trains a NeuralLM on a plain-text corpus (word-level tokens) and writes the
 * vocabulary + weights to disk.
 *
 * Usage:
 *   java -cp out com.chatllm.core.Trainer <corpus.txt> <output-model.bin> [epochs] [contextSize] [embedDim] [hiddenSize] [lr] [minFreq] [threads]
 *
 * Training is parallelized across CPU cores using the "Hogwild!" approach:
 * multiple threads run trainStep concurrently on the SAME shared weight arrays,
 * with no locking. That's a deliberate, well-known tradeoff — gradient updates
 * can occasionally clash between threads, making each individual update very
 * slightly noisier, but in practice this barely affects convergence for a
 * network this size, and it lets every CPU core actually pull its weight
 * instead of only one core ever being busy.
 */
public class Trainer {

    public static void main(String[] args) throws IOException, InterruptedException {
        if (args.length < 2) {
            System.out.println("Usage: Trainer <corpus.txt> <output-model.bin> [epochs] [contextSize] [embedDim] [hiddenSize] [lr] [minFreq] [threads]");
            System.out.println("Example: java -cp out com.chatllm.core.Trainer data/sample_corpus.txt model.bin 60 6 32 128 0.08 1");
            return;
        }
        String corpusPath = args[0];
        String outPath = args[1];
        int epochs = args.length > 2 ? Integer.parseInt(args[2]) : 60;
        int contextSize = args.length > 3 ? Integer.parseInt(args[3]) : 6;
        int embedDim = args.length > 4 ? Integer.parseInt(args[4]) : 32;
        int hiddenSize = args.length > 5 ? Integer.parseInt(args[5]) : 128;
        double lr = args.length > 6 ? Double.parseDouble(args[6]) : 0.08;
        int minFreq = args.length > 7 ? Integer.parseInt(args[7]) : 1;
        int threads = args.length > 8 ? Integer.parseInt(args[8]) : Runtime.getRuntime().availableProcessors();
        threads = Math.max(1, threads);

        String text = Files.readString(Paths.get(corpusPath));
        List<String> tokens = Vocabulary.tokenize(text);
        System.out.println("Corpus: " + text.length() + " characters -> " + tokens.size() + " word/punctuation tokens");

        Vocabulary vocab = Vocabulary.build(tokens, minFreq);
        System.out.println("Vocabulary size: " + vocab.size() + " unique tokens");
        System.out.println("Using " + threads + " CPU thread(s) for training "
                + "(" + Runtime.getRuntime().availableProcessors() + " available)");

        if (tokens.size() < contextSize + 10) {
            System.out.println("Corpus is too short for the chosen context size. Add more text and try again.");
            return;
        }

        int[] data = vocab.encode(tokens);
        NeuralLM model = new NeuralLM(vocab.size(), contextSize, embedDim, hiddenSize, 42);

        Integer[] positions = new Integer[data.length - contextSize];
        for (int i = 0; i < positions.length; i++) positions[i] = i + contextSize;
        Random shuffleRng = new Random(7);
        List<Integer> posList = Arrays.asList(positions);

        ExecutorService pool = threads > 1 ? Executors.newFixedThreadPool(threads) : null;

        long start = System.currentTimeMillis();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            Collections.shuffle(posList, shuffleRng);
            double totalLoss;

            if (pool == null) {
                totalLoss = runChunk(model, data, positions, 0, positions.length, contextSize, lr);
            } else {
                int chunkSize = (positions.length + threads - 1) / threads;
                List<Future<Double>> futures = new ArrayList<>();
                final double lrForEpoch = lr; // lambdas need an effectively-final capture
                for (int t = 0; t < threads; t++) {
                    int from = t * chunkSize;
                    int to = Math.min(from + chunkSize, positions.length);
                    if (from >= to) continue;
                    futures.add(pool.submit(() -> runChunk(model, data, positions, from, to, contextSize, lrForEpoch)));
                }
                totalLoss = 0;
                for (Future<Double> f : futures) {
                    try {
                        totalLoss += f.get();
                    } catch (ExecutionException e) {
                        throw new RuntimeException(e.getCause());
                    }
                }
            }

            double avg = totalLoss / positions.length;
            System.out.printf("Epoch %2d/%d  avg loss: %.4f  perplexity: %.2f  lr: %.4f%n",
                    epoch, epochs, avg, Math.exp(avg), lr);
            lr *= 0.97; // gentle decay so later epochs fine-tune instead of overshoot
        }
        if (pool != null) pool.shutdown();
        System.out.printf("Training finished in %.1fs%n", (System.currentTimeMillis() - start) / 1000.0);

        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(outPath)))) {
            vocab.save(out);
            model.save(out);
        }
        System.out.println("Model saved to " + outPath);
    }

    /** Runs trainStep over positions[from, to) on the shared model; returns the summed loss for that slice. */
    private static double runChunk(NeuralLM model, int[] data, Integer[] positions, int from, int to,
                                    int contextSize, double lr) {
        double loss = 0;
        for (int i = from; i < to; i++) {
            int p = positions[i];
            int[] ctx = Arrays.copyOfRange(data, p - contextSize, p);
            int target = data[p];
            loss += model.trainStep(ctx, target, lr);
        }
        return loss;
    }
}
