package com.chatllm.ui;

import com.chatllm.core.ChatEngine;

import java.nio.file.*;
import java.util.Scanner;

/**
 * A minimal frontend. It only knows about {@link ChatEngine} — it has no idea the
 * model underneath is a hand-rolled MLP, a transformer, or a remote API call.
 * Swap ChatEngine's implementation and this class doesn't need to change.
 */
public class ChatConsole {

    public static void main(String[] args) throws Exception {
        String modelPath = args.length > 0 ? args[0] : "model.bin";
        double temperature = args.length > 1 ? Double.parseDouble(args[1]) : 0.7;

        Path p = Paths.get(modelPath);
        if (!Files.exists(p)) {
            System.out.println("Model file not found: " + modelPath);
            System.out.println("Train one first, e.g.:");
            System.out.println("  java -cp out com.chatllm.core.Trainer data/sample_corpus.txt model.bin");
            return;
        }

        System.out.println("Loading model from " + modelPath + " ...");
        ChatEngine engine = ChatEngine.load(p, temperature);
        System.out.println("Ready! Type 'exit' to quit, 'reset' to clear conversation memory.\n");

        try (Scanner sc = new Scanner(System.in)) {
            while (true) {
                System.out.print("You: ");
                if (!sc.hasNextLine()) break;
                String line = sc.nextLine();
                if (line.equalsIgnoreCase("exit")) break;
                if (line.equalsIgnoreCase("reset")) {
                    engine.reset();
                    System.out.println("(conversation reset)");
                    continue;
                }
                String reply = engine.reply(line);
                System.out.println("Bot: " + (reply.isEmpty() ? "..." : reply));
            }
        }
        System.out.println("Bye!");
    }
}
