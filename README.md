# javallm — a tiny chat LLM in Java 17

A from-scratch, dependency-free neural language model you can train and chat with,
built to keep the **model** and the **frontend (UI)** cleanly separated.

## Honest expectations first

This is a **small, educational** language model — a single hidden-layer neural
network over a fixed window of previous characters (not a transformer, not
GPT-anything). With the tiny sample corpus included here it will produce
noticeably rough, sometimes gibberish text. That's expected, not a bug: real
chat-quality LLMs are trained on billions of words with GPU clusters for
weeks. What this project *does* give you is:

- A working, inspectable neural network (forward pass, backprop, gradient
  descent) implemented in plain Java — good for learning how these models work.
- A clean architecture split between "the model" and "the UI" that scales to
  a much bigger model later without touching the UI code.
- A real, if small-scale, training pipeline you can point at your own text.

If your goal is an actually good chat experience rather than an educational
exercise, see **"If you want real chat quality"** at the bottom — that section
points you to pretrained-model options instead of training from scratch.

## Architecture (model vs. UI, actually separate)

```
javallm/
├── core/   → com.chatllm.core     (THE MODEL — no UI code, no console/IO logic)
│    ├── Vocabulary.java   fixed char <-> integer mapping
│    ├── NeuralLM.java     the neural network: forward pass, backprop, save/load
│    ├── Trainer.java      training loop (reads a corpus, writes model.bin)
│    └── ChatEngine.java   PUBLIC API — the only class a frontend should call
│
├── ui/     → com.chatllm.ui       (THE FRONTEND — only depends on ChatEngine)
│    └── ChatConsole.java  a console chat loop
│
└── data/
     └── sample_corpus.txt small example dialogue corpus
```

The `ui` module only ever calls `ChatEngine.load(...)` and `engine.reply(...)`.
It has zero knowledge of embeddings, matrices, or training. That means you
could later:
- Swap `ChatEngine`'s internals for a bigger model, a different architecture,
  or even a call to a hosted API — `ChatConsole` wouldn't need to change.
- Build a completely different frontend (a Swing/JavaFX window, a REST
  controller, a Discord bot) against the same `ChatEngine` class.

## How the model works

- **Tokenizer**: word-level. `Vocabulary.tokenize` lowercases the text and
  splits it into word tokens (`[a-z0-9']+`) and single-character punctuation
  tokens, inserting a special `<nl>` token at every line break (this is what
  marked "end of a turn" back when tokens were characters, and still does).
  Unlike the fixed 96-character alphabet from before, the set of words is
  *learned from the training corpus*, so the vocabulary now has to be trained
  and saved/loaded together with the model — `model.bin` stores the
  vocabulary followed by the weights.
  - `<unk>` — any word not seen often enough in training falls back to this.
  - `<pad>` — filler used only to prime the very start of a conversation.
  - `<nl>` — end of a line/turn (used the way `'\n'` was used before).
- **Model**: for the last N tokens (`contextSize`, now measured in *words*,
  not characters), look up an embedding for each, concatenate them, pass
  through one `tanh` hidden layer, then a linear layer + softmax over every
  token in the vocabulary. `NeuralLM` itself didn't need to change at all —
  it was already generic over "however many symbols there are."
- **Training**: standard cross-entropy loss, plain SGD with manually derived
  gradients (no autograd library — every gradient in `NeuralLM.trainStep` is
  hand-derived calculus, which is why there are no external dependencies).
- **Chat**: your conversation is kept as a running list of token ids
  representing `user : <your message> <nl> bot :`. The model generates
  tokens one at a time (sampling from the softmax, with a temperature you
  control) until it produces `<nl>`, which is treated as "end of reply," then
  `Vocabulary.detokenize` joins the generated words back into a sentence
  (with a simple no-space-before-punctuation heuristic).

## Build

No Maven/Gradle needed — it's plain `javac`. Requires JDK 17+.

```bash
cd javallm
mkdir -p out
javac -encoding UTF-8 -d out core/src/main/java/com/chatllm/core/*.java
javac -encoding UTF-8 -d out -cp out ui/src/main/java/com/chatllm/ui/*.java
```

## Train

```bash
java -cp out com.chatllm.core.Trainer <corpus.txt> <output-model.bin> [epochs] [contextSize] [embedDim] [hiddenSize] [lr] [minFreq] [threads]
```

Using the bundled sample corpus:

```bash
java -cp out com.chatllm.core.Trainer data/sample_corpus.txt model.bin 60 6 32 128 0.08 1
```

Arguments (all but the first two are optional, shown with defaults):

| Arg | Meaning | Default |
|---|---|---|
| `corpus.txt` | plain text file to train on | required |
| `output-model.bin` | where to save vocabulary + weights | required |
| `epochs` | passes over the whole corpus | 60 |
| `contextSize` | how many previous **words/tokens** the model looks at | 6 |
| `embedDim` | size of each token's embedding vector | 32 |
| `hiddenSize` | size of the hidden layer | 128 |
| `lr` | initial learning rate (decays 3% per epoch) | 0.08 |
| `minFreq` | drop words seen fewer than this many times (become `<unk>`) | 1 |
| `threads` | CPU threads used for training (see below) | all available cores |

With the tiny bundled corpus the vocabulary ends up a few hundred words, and
the model can essentially memorize it (loss/perplexity will drop very low,
close to 1.0). That's expected at this scale — with a much bigger corpus,
`minFreq` (e.g. 2–5) and a larger `hiddenSize`/`embedDim` become useful to
keep the model from either exploding in vocabulary size or overfitting.

Watch the printed `avg loss` / `perplexity` — perplexity should trend down.
It's normal for loss to fluctuate for the first several epochs before it
settles, especially at higher learning rates.

### Why training can be slow, and how to speed it up

If training takes many minutes, it's almost never the GPU's fault (this
project has no GPU code path at all — see the Colab section below). The two
real costs are:

1. **It was single-threaded.** Every `trainStep` touches the full output
   layer (`hiddenSize × vocabSize` numbers, both forward and backward), and
   the original code did this one token at a time on one CPU core, so a
   multi-core machine was mostly sitting idle. `Trainer` now parallelizes
   across CPU cores automatically (defaults to `Runtime.getRuntime()
   .availableProcessors()`, or pass a `threads` argument to control it
   yourself), using the standard "Hogwild!" trick: multiple threads run
   `trainStep` concurrently on the *same* shared weight arrays with no
   locking. Updates can occasionally clash between threads, which adds a
   tiny bit of noise to individual gradient steps, but in practice this
   barely affects convergence for a network this small — `Trainer` prints
   how many threads it's using and how many are available.
2. **Vocabulary size and hidden size dominate.** A bigger corpus → a bigger
   vocabulary → a bigger, slower output layer. If a bigger corpus is still
   too slow even with more threads, raise `minFreq` (e.g. to 3-5) to cut the
   vocabulary down, and/or lower `hiddenSize`.

## Train faster: Google Colab

You can absolutely run this in Colab for free extra compute — but pick a
**CPU runtime**, not GPU. This project has zero GPU code (plain Java
`double[][]` loops, no CUDA/JCuda/DL4J), so a T4 will sit completely idle;
selecting GPU just wastes a scarce free-tier GPU allocation for nothing.
Colab's CPU, combined with the multi-threading above, is the actual speedup —
though note the free tier is usually only ~2 vCPUs, so don't expect a huge
multi-core win either.

**Cell 1 — Install Java 17**
```python
!apt-get update -qq
!apt-get install -y -qq openjdk-17-jdk-headless
!java -version
```

**Cell 2 — Upload your project zip**
```python
from google.colab import files
uploaded = files.upload()   # choose javallm.zip
```

**Cell 3 — Unzip and enter the project folder**
```python
!unzip -o -q javallm.zip -d /content
%cd /content/javallm
```

**Cell 4 — Compile**
```python
!mkdir -p out
!javac -encoding UTF-8 -d out core/src/main/java/com/chatllm/core/*.java
!javac -encoding UTF-8 -d out -cp out ui/src/main/java/com/chatllm/ui/*.java
```

**Cell 5 (optional) — Upload your own corpus**
```python
from google.colab import files
uploaded = files.upload()          # choose your .txt corpus file
import os
os.rename(list(uploaded.keys())[0], "data/my_corpus.txt")
```

**Cell 6 — Check available cores, then train**
```python
!nproc
!java -cp out com.chatllm.core.Trainer data/sample_corpus.txt model.bin 60 6 32 128 0.08 1
```

**Cell 7 — Download the trained model**
```python
from google.colab import files
files.download("model.bin")
```

If you want the T4 to actually matter, that requires a different foundation
entirely — a GPU-backed library (DJL, ND4J-CUDA) or a PyTorch reimplementation
— not just running this Java code in a GPU-enabled notebook. Happy to build
that version if you want to go there.

## Chat

```bash
java -cp out com.chatllm.ui.ChatConsole model.bin 0.7
```

- First argument: path to your trained `model.bin`.
- Second argument: sampling temperature (0.0–1.5ish). Lower = safer/more
  repetitive, higher = more random/creative (and more likely to be garbled
  with a model this small). Try 0.5–0.8.
- Type `reset` to clear the conversation and start fresh, `exit` to quit.

## How to actually get better results from this codebase

The single biggest lever is **training data**. `data/sample_corpus.txt` has
~60 short exchanges — enough for the model to essentially memorize answers to
those exact questions, but it won't generalize to phrasing it hasn't seen. To
improve it:

1. **Write a much bigger corpus** in the same `User: ...\nBot: ...\n\n` format
   — hundreds or thousands of exchanges, and vary the phrasing of similar
   questions so the model sees more than one way to ask the same thing. Keep
   the format consistent since the model has to *learn* the "User: / Bot:"
   turn-taking pattern itself.
2. **Increase capacity along with data**: bigger corpus → try
   `contextSize=10-15`, `embedDim=64-128`, `hiddenSize=256-512`.
3. **Raise `minFreq`** (e.g. 2–5) once your corpus is large, so one-off typos
   and rare words don't bloat the vocabulary — they'll map to `<unk>` instead.
4. **Train longer** and watch perplexity — stop increasing epochs once it
   plateaus (or creeps back up, which means overfitting on small data).

Even maxed out, this architecture (a fixed-window MLP) has a hard ceiling —
it has no real memory of anything outside its context window and no attention
mechanism, so it will never produce GPT-quality conversation. That's the
tradeoff for having something you can fully read, train, and understand in a
few hundred lines of plain Java.

## AFK mode: two bots talking to (and training) themselves

```bash
java -cp out com.chatllm.ui.SelfChatConsole model.bin [temperature] [learnRate] [delayMs] [seedMessage] [maxTurns]
```

This loads `model.bin` **twice** into two fully independent `ChatEngine`
instances — "Bot" and "Bot2", each with its own copy of the weights — and
lets them talk to each other indefinitely (or for `maxTurns` exchanges, `0` =
forever). Defaults: `temperature=0.8`, `learnRate=0.02`, `delayMs=600`,
`seedMessage="hello"`.

While they talk, **both bots keep training live** on the conversation via
`ChatEngine.learnFromHistory` — same idea as `Trainer`, just applied online
to what's actually being said instead of a fixed corpus file. You can jump in
any time by typing a line (redirects the conversation), `reset` clears both
bots' memory, and `exit` stops the loop and saves each bot's now-live-trained
weights to `bot_trained.bin` / `bot2_trained.bin`.

Two honest caveats worth knowing before you walk away from this for an hour:

- **There's no external "correct answer" here.** `learnFromHistory` just
  reinforces whatever pattern actually occurred, good or bad — it's not
  reward-based or judged by anything. It cannot make the bots smarter; it can
  only make them more confident in whatever they already tend to say.
- **This tends toward repetition, not improvement.** Training a model on its
  own generated output is a known way to induce "mode collapse" — you'll
  likely watch coherent replies (borrowed from the original training corpus)
  gradually degrade into loops and garbled fragments the longer it runs,
  especially at the small scale of this demo model. That's expected, not a
  bug — it's a good illustration of exactly why real LLM training uses
  curated data and human/automated feedback rather than raw self-play.

## If you want real chat quality

Training a genuinely good LLM from scratch requires far more data and compute
than a laptop can realistically provide. If actual chat quality (not the
learning exercise) is the goal, better paths in the Java ecosystem are:

- **Run an existing open-weights model from Java**: libraries like
  [Deep Java Library (DJL)](https://djl.ai/) or JNI/JNA bindings to
  [llama.cpp](https://github.com/ggml-org/llama.cpp) let you load a pretrained
  model (e.g. Llama, Mistral, Gemma) and serve it from a Java application —
  you get real conversational quality without training anything yourself.
- **Fine-tune** a small pretrained open model on your own data (via Python/
  PyTorch or Hugging Face tooling) and then serve the result from Java through
  one of the above, if you want it specialized to your own domain.
- **Call a hosted API** (such as Anthropic's Claude API) from your Java
  backend if you want state-of-the-art quality with no training or local
  model-serving at all — `ChatEngine` in this project is exactly the seam
  you'd swap out for an HTTP call to an API in that case.

This project deliberately does the "from scratch" version so you can see and
understand every line of math involved, but that transparency is the tradeoff
against the quality those larger approaches would give you.
