package com.chatllm.core;

import java.io.*;
import java.util.*;
import java.util.regex.*;

/**
 * Word-level vocabulary and tokenizer.
 *
 * Unlike a fixed character set, the set of words is learned from the training
 * corpus, so (unlike before) the vocabulary now has to be saved and loaded
 * together with the model weights — see Trainer.save and ChatEngine.load.
 */
public class Vocabulary {

    public static final String UNK = "<unk>"; // unknown / out-of-vocabulary word
    public static final String PAD = "<pad>"; // padding used before any real conversation
    public static final String NL  = "<nl>";  // marks a line break / end of a turn

    // A "word" is letters/digits/apostrophes glued together; anything else
    // (punctuation, symbols) becomes its own single-character token.
    private static final Pattern TOKEN_PATTERN = Pattern.compile("[a-z0-9']+|[^\\sa-z0-9']");

    private final List<String> idToToken;
    private final Map<String, Integer> tokenToId;
    public final int unkId;
    public final int padId;
    public final int nlId;

    private Vocabulary(List<String> idToToken) {
        this.idToToken = idToToken;
        this.tokenToId = new HashMap<>();
        for (int i = 0; i < idToToken.size(); i++) tokenToId.put(idToToken.get(i), i);
        this.unkId = tokenToId.get(UNK);
        this.padId = tokenToId.get(PAD);
        this.nlId = tokenToId.get(NL);
    }

    /** Splits raw text into word/punctuation tokens, inserting NL between lines. */
    public static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        String[] lines = text.toLowerCase(Locale.ROOT).split("\n", -1);
        for (int i = 0; i < lines.length; i++) {
            Matcher m = TOKEN_PATTERN.matcher(lines[i]);
            while (m.find()) tokens.add(m.group());
            if (i < lines.length - 1) tokens.add(NL);
        }
        return tokens;
    }

    /** Builds a vocabulary from a token stream. Words seen fewer than minFreq times become UNK. */
    public static Vocabulary build(List<String> tokens, int minFreq) {
        Map<String, Integer> freq = new LinkedHashMap<>();
        for (String t : tokens) freq.merge(t, 1, Integer::sum);

        List<String> vocab = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (String special : new String[] { UNK, PAD, NL }) {
            vocab.add(special);
            seen.add(special);
        }
        for (Map.Entry<String, Integer> e : freq.entrySet()) {
            String tok = e.getKey();
            if (seen.contains(tok)) continue;
            if (e.getValue() >= minFreq) {
                vocab.add(tok);
                seen.add(tok);
            }
        }
        return new Vocabulary(vocab);
    }

    public int size() { return idToToken.size(); }

    public int idOf(String token) {
        return tokenToId.getOrDefault(token, unkId);
    }

    public String tokenOf(int id) {
        if (id < 0 || id >= idToToken.size()) return UNK;
        return idToToken.get(id);
    }

    public int[] encode(List<String> tokens) {
        int[] out = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) out[i] = idOf(tokens.get(i));
        return out;
    }

    /** Turns generated tokens back into human-readable text (simple spacing heuristic). */
    public String detokenize(List<String> tokens) {
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (String t : tokens) {
            if (t.equals(NL) || t.equals(PAD) || t.equals(UNK)) continue;
            boolean noSpaceBefore = t.length() == 1 && ".,!?;:')]}".indexOf(t.charAt(0)) >= 0;
            if (!first && !noSpaceBefore) sb.append(' ');
            sb.append(t);
            first = false;
        }
        return sb.toString();
    }

    // ---------------------------------------------------------------- persistence

    public void save(DataOutputStream out) throws IOException {
        out.writeInt(idToToken.size());
        for (String t : idToToken) out.writeUTF(t);
    }

    public static Vocabulary load(DataInputStream in) throws IOException {
        int n = in.readInt();
        List<String> tokens = new ArrayList<>(n);
        for (int i = 0; i < n; i++) tokens.add(in.readUTF());
        return new Vocabulary(tokens);
    }
}
