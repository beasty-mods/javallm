package com.chatllm.core;

import java.io.*;
import java.util.Random;

/**
 * A small character-level neural language model:
 *
 *   context chars --embedding--> concat --W1,tanh--> hidden --W2,softmax--> next-char distribution
 *
 * It is a fixed-window ("n-gram style") model rather than a recurrent or transformer
 * network, which keeps the math simple enough to implement (and verify) as plain
 * hand-written forward/backward passes with no external ML library.
 */
public class NeuralLM {

    public final int vocabSize;
    public final int contextSize;
    public final int embedDim;
    public final int hiddenSize;
    private final int inputSize; // contextSize * embedDim

    double[][] embedding; // [vocabSize][embedDim]
    double[][] W1;        // [inputSize][hiddenSize]
    double[] b1;          // [hiddenSize]
    double[][] W2;        // [hiddenSize][vocabSize]
    double[] b2;          // [vocabSize]

    public NeuralLM(int vocabSize, int contextSize, int embedDim, int hiddenSize, long seed) {
        this.vocabSize = vocabSize;
        this.contextSize = contextSize;
        this.embedDim = embedDim;
        this.hiddenSize = hiddenSize;
        this.inputSize = contextSize * embedDim;

        Random r = new Random(seed);
        embedding = randMatrix(vocabSize, embedDim, r, 1.0 / Math.sqrt(embedDim));
        W1 = randMatrix(inputSize, hiddenSize, r, 1.0 / Math.sqrt(inputSize));
        b1 = new double[hiddenSize];
        W2 = randMatrix(hiddenSize, vocabSize, r, 1.0 / Math.sqrt(hiddenSize));
        b2 = new double[vocabSize];
    }

    /** Constructor used only when loading weights from disk. */
    private NeuralLM(int vocabSize, int contextSize, int embedDim, int hiddenSize) {
        this.vocabSize = vocabSize;
        this.contextSize = contextSize;
        this.embedDim = embedDim;
        this.hiddenSize = hiddenSize;
        this.inputSize = contextSize * embedDim;
    }

    private static double[][] randMatrix(int rows, int cols, Random r, double scale) {
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = (r.nextDouble() * 2 - 1) * scale;
        return m;
    }

    public static final class ForwardCache {
        int[] contextIdx;
        double[] x;      // concatenated context embeddings, size inputSize
        double[] h;      // hidden activations post-tanh, size hiddenSize
        double[] logits; // size vocabSize
        double[] probs;  // softmax over logits, size vocabSize
    }

    public ForwardCache forward(int[] contextIdx) {
        ForwardCache c = new ForwardCache();
        c.contextIdx = contextIdx;

        double[] x = new double[inputSize];
        for (int t = 0; t < contextSize; t++) {
            double[] e = embedding[contextIdx[t]];
            System.arraycopy(e, 0, x, t * embedDim, embedDim);
        }
        c.x = x;

        double[] h = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double s = b1[j];
            for (int i = 0; i < inputSize; i++) s += x[i] * W1[i][j];
            h[j] = Math.tanh(s);
        }
        c.h = h;

        double[] logits = new double[vocabSize];
        for (int k = 0; k < vocabSize; k++) {
            double s = b2[k];
            for (int j = 0; j < hiddenSize; j++) s += h[j] * W2[j][k];
            logits[k] = s;
        }
        c.logits = logits;
        c.probs = softmax(logits, 1.0);
        return c;
    }

    private static double[] softmax(double[] logits, double temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logits) if (v > max) max = v;
        double sum = 0;
        double[] out = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            out[i] = Math.exp((logits[i] - max) / temperature);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) out[i] /= sum;
        return out;
    }

    /** One SGD step on a single (context, target) example. Returns the cross-entropy loss. */
    public double trainStep(int[] contextIdx, int targetIdx, double lr) {
        ForwardCache c = forward(contextIdx);
        double loss = -Math.log(Math.max(c.probs[targetIdx], 1e-12));

        double[] dLogits = c.probs.clone();
        dLogits[targetIdx] -= 1.0;

        double[] dh = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double sum = 0;
            for (int k = 0; k < vocabSize; k++) sum += dLogits[k] * W2[j][k];
            dh[j] = sum * (1 - c.h[j] * c.h[j]); // tanh derivative
        }

        for (int j = 0; j < hiddenSize; j++)
            for (int k = 0; k < vocabSize; k++)
                W2[j][k] -= lr * dLogits[k] * c.h[j];
        for (int k = 0; k < vocabSize; k++) b2[k] -= lr * dLogits[k];

        double[] dx = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            double sum = 0;
            for (int j = 0; j < hiddenSize; j++) sum += dh[j] * W1[i][j];
            dx[i] = sum;
        }
        for (int i = 0; i < inputSize; i++)
            for (int j = 0; j < hiddenSize; j++)
                W1[i][j] -= lr * dh[j] * c.x[i];
        for (int j = 0; j < hiddenSize; j++) b1[j] -= lr * dh[j];

        for (int t = 0; t < contextSize; t++) {
            double[] emb = embedding[contextIdx[t]];
            int base = t * embedDim;
            for (int d = 0; d < embedDim; d++) emb[d] -= lr * dx[base + d];
        }

        return loss;
    }

    /** Samples the next character index given a context window, using temperature-scaled softmax. */
    public int sampleNext(int[] contextIdx, double temperature, Random rng) {
        ForwardCache c = forward(contextIdx);
        double[] probs = softmax(c.logits, Math.max(temperature, 1e-6));
        double r = rng.nextDouble();
        double cum = 0;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r <= cum) return i;
        }
        return probs.length - 1;
    }

    // ---------------------------------------------------------------- persistence

    public void save(DataOutputStream out) throws IOException {
        out.writeInt(vocabSize);
        out.writeInt(contextSize);
        out.writeInt(embedDim);
        out.writeInt(hiddenSize);
        writeMatrix(out, embedding);
        writeMatrix(out, W1);
        writeVector(out, b1);
        writeMatrix(out, W2);
        writeVector(out, b2);
    }

    public static NeuralLM load(DataInputStream in) throws IOException {
        int vocabSize = in.readInt();
        int contextSize = in.readInt();
        int embedDim = in.readInt();
        int hiddenSize = in.readInt();
        NeuralLM m = new NeuralLM(vocabSize, contextSize, embedDim, hiddenSize);
        m.embedding = readMatrix(in, vocabSize, embedDim);
        m.W1 = readMatrix(in, contextSize * embedDim, hiddenSize);
        m.b1 = readVector(in, hiddenSize);
        m.W2 = readMatrix(in, hiddenSize, vocabSize);
        m.b2 = readVector(in, vocabSize);
        return m;
    }

    private static void writeMatrix(DataOutputStream out, double[][] m) throws IOException {
        for (double[] row : m) for (double v : row) out.writeDouble(v);
    }

    private static void writeVector(DataOutputStream out, double[] v) throws IOException {
        for (double d : v) out.writeDouble(d);
    }

    private static double[][] readMatrix(DataInputStream in, int rows, int cols) throws IOException {
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) m[i][j] = in.readDouble();
        return m;
    }

    private static double[] readVector(DataInputStream in, int n) throws IOException {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = in.readDouble();
        return v;
    }
}
