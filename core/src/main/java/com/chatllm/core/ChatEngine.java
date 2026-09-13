package com.chatllm.core;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Public API surface of the model. A UI only needs {@link #load} and {@link #reply};
 * it never touches NeuralLM, Vocabulary, or Trainer directly. This is the seam that
 * keeps "the model" and "the frontend" separate: swap this class's implementation
 * (a bigger model, a remote API call, etc.) and no UI code needs to change.
 *
 * Conversation state is kept as a running list of word/punctuation token ids
 * (word-level, not character-level).
 */
public class ChatEngine {

    private static final int MAX_HISTORY_TOKENS = 800;
    private static final int MAX_REPLY_TOKENS = 40;

    private final NeuralLM model;
    private final Vocabulary vocab;
    private final List<Integer> history;
    private final Random rng;
    private final double temperature;
    private int trainedUpTo; // history index already consumed by learnFromHistory

    private ChatEngine(NeuralLM model, Vocabulary vocab, double temperature, long seed) {
        this.model = model;
        this.vocab = vocab;
        this.temperature = temperature;
        this.rng = new Random(seed);
        this.history = new ArrayList<>();
        for (int i = 0; i < model.contextSize; i++) history.add(vocab.padId); // seed padding
        this.trainedUpTo = history.size();
    }

    public static ChatEngine load(Path modelFile, double temperature) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(modelFile)))) {
            Vocabulary vocab = Vocabulary.load(in);
            NeuralLM model = NeuralLM.load(in);
            return new ChatEngine(model, vocab, temperature, System.nanoTime());
        }
    }

    /** Clears conversational memory but keeps the loaded weights. */
    public void reset() {
        history.clear();
        for (int i = 0; i < model.contextSize; i++) history.add(vocab.padId);
        trainedUpTo = history.size();
    }

    /** Feeds the user's message to the model and returns its generated reply. */
    public String reply(String userMessage) {
        appendUserTurn(userMessage);
        return generateReply(this.temperature);
    }

    /** Appends a user turn ("user : ... <nl> bot :") to history without generating a reply yet. */
    public void appendUserTurn(String userMessage) {
        List<String> userTurn = new ArrayList<>();
        userTurn.add("user");
        userTurn.add(":");
        userTurn.addAll(Vocabulary.tokenize(userMessage));
        userTurn.add(Vocabulary.NL);
        userTurn.add("bot");
        userTurn.add(":");
        for (String t : userTurn) history.add(vocab.idOf(t));
    }

    /** Current history length — pair with {@link #rollbackTo} to discard a generation attempt. */
    public int historyMark() {
        return history.size();
    }

    /** Discards everything generated after the given mark, e.g. to retry a repetitive reply. */
    public void rollbackTo(int mark) {
        if (mark < history.size()) history.subList(mark, history.size()).clear();
    }

    /**
     * Generates a reply from the current history using the given sampling temperature,
     * without appending a new user turn first. Can be called again after
     * {@link #rollbackTo} to regenerate the same turn (e.g. with a higher temperature
     * to break out of a repetitive loop).
     */
    public String generateReply(double temperatureOverride) {
        List<String> replyTokens = new ArrayList<>();
        int cs = model.contextSize;
        for (int step = 0; step < MAX_REPLY_TOKENS; step++) {
            int histLen = history.size();
            int[] ctx = new int[cs];
            for (int i = 0; i < cs; i++) ctx[i] = history.get(histLen - cs + i);

            int nextId = model.sampleNext(ctx, temperatureOverride, rng);
            if (nextId == vocab.nlId) break;

            replyTokens.add(vocab.tokenOf(nextId));
            history.add(nextId);
        }
        history.add(vocab.nlId);
        trimHistory();

        return vocab.detokenize(replyTokens);
    }

    private void trimHistory() {
        if (history.size() > MAX_HISTORY_TOKENS) {
            int removeCount = history.size() - MAX_HISTORY_TOKENS;
            history.subList(0, removeCount).clear();
            trainedUpTo = Math.max(0, trainedUpTo - removeCount);
        }
    }

    /**
     * Runs a few live SGD steps using the conversation so far as its own training
     * data: "given these previous tokens, predict the token that actually came
     * next" — exactly like {@code Trainer}, just applied online to whatever has
     * been said instead of to a fixed corpus file.
     * <p>
     * Be clear-eyed about what this is and isn't: there's no external judge of
     * "good" or "correct" here, so this cannot make the bot smarter in any
     * meaningful sense. It only reinforces whatever patterns already showed up
     * in the conversation — which, left running long enough, tends to make output
     * more repetitive and stuck in a rut (a well-known failure mode of training a
     * model on its own generations) rather than more capable.
     */
    public void learnFromHistory(double lr) {
        int cs = model.contextSize;
        int from = Math.max(trainedUpTo, cs);
        for (int p = from; p < history.size(); p++) {
            int target = history.get(p);
            if (target == vocab.padId) continue;
            int[] ctx = new int[cs];
            for (int i = 0; i < cs; i++) ctx[i] = history.get(p - cs + i);
            model.trainStep(ctx, target, lr);
        }
        trainedUpTo = history.size();
    }

    /** Saves this engine's current vocabulary + (possibly live-trained) weights. */
    public void save(Path outFile) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(outFile)))) {
            vocab.save(out);
            model.save(out);
        }
    }
}
