package com.chatllm.ui;

import com.chatllm.core.ChatEngine;

import java.io.*;
import java.nio.file.*;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * "AFK mode": loads the same trained model TWICE — once as "Bot" and once as
 * "Bot2" — two fully independent ChatEngine instances, each with its own copy
 * of the weights and its own conversation memory. They talk to each other
 * while you watch (or step away), and both keep training themselves live on
 * whatever gets said, via {@link ChatEngine#learnFromHistory}.
 * <p>
 * Because each copy trains only on what it personally says/hears, and starts
 * sampling from the same weights but with different randomness, the two will
 * gradually drift apart the longer they talk — hence "Bot" vs "Bot2".
 * <p>
 * You can jump in any time: type a line and press Enter to redirect the
 * conversation instead of letting the bots continue on their own. Type
 * 'reset' to clear both bots' memory, 'exit' to stop (weights get saved on
 * exit so the live training isn't lost).
 * <p>
 * This class only calls the same public ChatEngine methods ChatConsole does
 * (plus learnFromHistory/save) — it still knows nothing about embeddings,
 * matrices, or tokenization.
 */
public class SelfChatConsole {

    private static final int REPEAT_WINDOW = 4;      // how many recent replies per speaker to compare against
    private static final int MAX_REGEN_ATTEMPTS = 3;  // how many times to retry a repetitive reply
    private static final double TEMP_BOOST_STEP = 0.35;
    private static final double MAX_TEMP = 1.8;

    public static void main(String[] args) throws Exception {
        String modelPath = args.length > 0 ? args[0] : "model.bin";
        // Default > 1.0 on purpose: this demo model is small and easily overfit (near-deterministic
        // predictions), so a temperature BELOW 1.0 sharpens it further and makes loops more likely,
        // not less. Values above 1.0 flatten the distribution and add real variety.
        double temperature = args.length > 1 ? Double.parseDouble(args[1]) : 1.1;
        double learnRate = args.length > 2 ? Double.parseDouble(args[2]) : 0.02;
        int delayMillis = args.length > 3 ? Integer.parseInt(args[3]) : 600;
        String seedMessage = args.length > 4 ? args[4] : "hello";
        int maxTurns = args.length > 5 ? Integer.parseInt(args[5]) : 0; // 0 = run forever

        Path modelFile = Paths.get(modelPath);
        if (!Files.exists(modelFile)) {
            System.out.println("Model file not found: " + modelPath);
            System.out.println("Train one first, e.g.:");
            System.out.println("  java -cp out com.chatllm.core.Trainer data/sample_corpus.txt model.bin");
            return;
        }

        System.out.println("Loading the model twice: 'Bot' and 'Bot2' (independent copies) ...");
        ChatEngine bot = ChatEngine.load(modelFile, temperature);
        ChatEngine bot2 = ChatEngine.load(modelFile, temperature);

        System.out.println("They'll now chat with each other and keep training live on the conversation.");
        System.out.println("If a bot starts repeating itself, it'll notice and shake things up automatically.");
        System.out.println("Type anytime to jump in, 'reset' to clear memory, 'exit' to stop.\n");

        BlockingQueue<String> userInput = new LinkedBlockingQueue<>();
        Thread reader = new Thread(() -> {
            try (Scanner sc = new Scanner(System.in)) {
                while (sc.hasNextLine()) userInput.put(sc.nextLine());
            } catch (InterruptedException ignored) {
                // program is shutting down
            }
        });
        reader.setDaemon(true);
        reader.start();

        Map<String, Deque<String>> recentBySpeaker = new HashMap<>();
        recentBySpeaker.put("Bot", new ArrayDeque<>());
        recentBySpeaker.put("Bot2", new ArrayDeque<>());

        String message = seedMessage;
        boolean addressingBot = true; // whose turn it is to receive `message`
        boolean running = true;
        int turnCount = 0;

        while (running) {
            String injected = userInput.poll();
            if (injected != null) {
                if (injected.equalsIgnoreCase("exit")) break;
                if (injected.equalsIgnoreCase("reset")) {
                    bot.reset();
                    bot2.reset();
                    recentBySpeaker.get("Bot").clear();
                    recentBySpeaker.get("Bot2").clear();
                    System.out.println("(both bots' conversation memory reset)");
                    continue;
                }
                System.out.println("You: " + injected);
                message = injected; // redirect the conversation toward whoever's turn is next
            }

            ChatEngine speaker = addressingBot ? bot : bot2;
            String label = addressingBot ? "Bot" : "Bot2";
            Deque<String> recent = recentBySpeaker.get(label);

            speaker.appendUserTurn(message);
            int mark = speaker.historyMark();

            double temp = temperature;
            String reply = speaker.generateReply(temp);

            int attempts = 0;
            while (isRepeat(reply, recent) && attempts < MAX_REGEN_ATTEMPTS) {
                attempts++;
                speaker.rollbackTo(mark);
                temp = Math.min(temp + TEMP_BOOST_STEP, MAX_TEMP);
                reply = speaker.generateReply(temp);
            }
            if (attempts > 0) {
                boolean fixed = !isRepeat(reply, recent);
                System.out.println("  (" + label + " noticed it was repeating itself, "
                        + (fixed ? "shook things up" : "tried to shake things up") + " — temp -> "
                        + String.format("%.2f", temp) + ")");
            }

            recent.addLast(normalize(reply));
            while (recent.size() > REPEAT_WINDOW) recent.removeFirst();

            speaker.learnFromHistory(learnRate); // live self-training on the final (non-looping) exchange

            System.out.println(label + ": " + (reply.isEmpty() ? "..." : reply));

            message = reply;
            addressingBot = !addressingBot;
            turnCount++;
            if (maxTurns > 0 && turnCount >= maxTurns) break;

            Thread.sleep(delayMillis);
        }

        System.out.println("\nStopping. Saving each bot's live-trained weights...");
        bot.save(Paths.get("bot_trained.bin"));
        bot2.save(Paths.get("bot2_trained.bin"));
        System.out.println("Saved bot_trained.bin and bot2_trained.bin");
    }

    private static boolean isRepeat(String reply, Deque<String> recent) {
        return recent.contains(normalize(reply));
    }

    private static String normalize(String s) {
        return s.trim().toLowerCase();
    }
}
