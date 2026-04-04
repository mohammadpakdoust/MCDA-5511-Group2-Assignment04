# Mechanistic Interpretability with Sparse Autoencoders (SAEs)

> **MCDA-5511 · Assignment 04 · Group 2**
> Probing the residual stream of `Qwen2.5-1.5B` with a trained Sparse Autoencoder to discover and causally test interpretable features.

**Team Members:**
- Mohammad Pakdoust
- MD Musfiqur Rahman
- Krushi Mistry

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Methodology](#2-methodology)
3. [Experiments](#3-experiments)
4. [Feature Analysis](#4-feature-analysis)
5. [Automated Interpretability](#5-automated-interpretability)
6. [Causal Intervention](#6-causal-intervention)
7. [Challenges & Limitations](#7-challenges--limitations)
8. [Insights & Discussion](#8-insights--discussion)

---

## 📋 Grading Rubric Mapping

To assist with grading, the following table maps the specific assignment requirements to their corresponding sections within this report:

| Requirement | Corresponding Section |
|---|---|
| **Question 1:** Modular code design and pipeline | [2. Methodology](#2-methodology-) |
| **Question 2:** Dataset generating, prompt selection, stats | [3. Experiments (Corpus Selection)](#corpus-selection--statistics-) |
| **Question 3:** SAE framework, tie-weights, early stopping, sparsity | [1. Introduction](#1-introduction-) & [3.3 Verification](#33-verification-of-learned-sparsity) |
| **Question 4:** Manual feature extraction & interpretation | [4. Feature Analysis](#4-feature-analysis-) |
| **Question 5 (Bonus):** Automated machine learning clustering | [5. Automated Interpretability](#5-automated-interpretability-) |
| **Question 6:** Intervention / counterfactual clamping | [6. Causal Intervention](#6-causal-intervention-) |

---

## 1. Introduction 📌 *(Answers Q3)*

### What is Activation Space?

In a transformer model the "activation space" refers to the high-dimensional vector representations of tokens at a given layer. These vectors (the **residual stream**) encode all information the model uses for computation, but they are dense and difficult for humans to interpret directly because of **superposition** — the phenomenon where the model packs more *features* into fewer dimensions than it has neurons.

### What is a Feature?

Anthropic defines a **feature** as a semantically coherent direction in activation space. A single neuron can be *polysemantic* (responding to multiple unrelated concepts), but an SAE identifies *monosemantic* directions that each correspond to one clear concept (e.g. "French grammar", "medical terminology", or "sentiment").

### How SAEs Map Activations to Features

An SAE is trained to reconstruct activations $x$ through a high-dimensional, sparse bottleneck $f$:

1. **Encoder** — Projects activations into a higher-dimensional space (4× to 64× the original) and applies ReLU to enforce non-negative sparsity.
2. **Sparsity Penalty** — An $L_1$ regularisation term in the loss pushes most features to zero, ensuring only a small number are active for any given token.
3. **Decoder** — Reconstructs the original activation from the sparse feature vector.

$$x \approx \hat{x} = W_{\text{dec}} \cdot \text{ReLU}\!\left(W_{\text{enc}}(x - b_{\text{dec}}) + b_{\text{enc}}\right) + b_{\text{dec}}$$

### Hook Placement Rationale

For this assignment, we inserted PyTorch forward hooks to capture the residual stream of **Layer 14** (the middle layer of the 28-layer `Qwen2.5-1.5B` model). Our reasoning closely follows the rationale found in Anthropic's research, specifically *"Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"*.

Anthropic highlights that different transformers layers build conceptually different representations:
- **Early layers** predominantly encode local syntax, raw spelling, and simple surface-level grammar.
- **Late layers** are heavily geared towards output preparation for the final logits.
- **Middle layers**, however, tend to exhibit rich, abstract representations of world knowledge, multi-token concepts, entity types, and deep semantic structures. 

By hooking into the middle layer's residual stream, we maximize our chances of extracting high-level, interpretable features (such as tracking temporal markers, discourse structures, or conceptual entities) rather than just shallow syntactic or output-oriented features. 

---

## 2. Methodology 📌 *(Answers Q1)*

### Project Structure

```text
sae-interpretability/
├── main.py                  # Central execution pipeline
├── models/
│   └── model_wrapper.py     # HuggingFace model loading & hook management
├── data/
│   └── dataset_generator.py # Activation extraction from WikiText-2
├── sae/
│   ├── sae_model.py         # PyTorch SAE implementation
│   └── train_sae.py         # Training loop, validation, early stopping
├── features/
│   ├── feature_extractor.py # Top-k activating example retrieval
│   ├── feature_analyzer.py  # Interpretation and summary tools
│   └── intervention.py      # Causal feature clamping via forward hooks
├── utils/
│   └── helpers.py           # Visualisation & logging utilities
└── pyproject.toml           # Dependency management (uv)
```

### Experimental Setup

| Parameter | Value |
|---|---|
| **Base model** | `Qwen/Qwen2.5-1.5B` |
| **Layer probed** | Residual stream output of Layer 14 (mid-network) |
| **Hidden dim** | 1 536 |
| **SAE feature dim** | 6 144 (4× expansion) |
| **Training data** | WikiText-2-raw-v1 (500 samples, max 128 tokens each) |
| **Batch size** | 128 |
| **Optimiser** | Adam · lr = 1e-4 |
| **Epochs** | 20 (with early stopping, patience = 5) |
| **L1 regularisation (λ)** | 0.1 (default) |
| **Hardware** | Apple Silicon (MPS / float16) |

### Corpus Selection & Statistics 📌 *(Answers Q2)*

For our training dataset, we utilized the `WikiText-2-raw-v1` corpus sourced directly via the Hugging Face `datasets` library. 

**Why this corpus?** We specifically chose Wikipedia article excerpts because they provide a rich, dense blend of factual, historical, and biographical prose. Rather than focusing on highly specialized topics (like purely legal or medical text) which might lead to overfitting on niche jargon, encyclopedic text offers a broad baseline of general knowledge, syntactic variety, and named entities. This makes it an ideal testing ground for identifying broad thematic features (like geographical nouns, syntax boundaries, and numeric/factual quantifiers).

**Summary Statistics of the Generated Data:**
- **Source Corpus**: `WikiText-2-raw-v1`
- **Number of Prompts/Passages Processed**: 500
- **Average Tokens per Passage**: ~96
- **Maximum Tokens per Passage**: 128
- **Total Token Activations Extracted**: ~48,000

### Setup & Usage

This project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible environment management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

#### Run the Full Pipeline

```bash
# With the venv activated:
python main.py --max_samples 500 --epochs 20

# Or run directly via uv without activating:
uv run python main.py --max_samples 500 --epochs 20
```

This will:
1. Load `Qwen/Qwen2.5-1.5B`.
2. Extract Layer-14 residual-stream activations from WikiText-2 (cached to `data/activations_layer_14.pt`).
3. Train the SAE and save the best checkpoint.
4. Print the top-activating token contexts for the five most prominent features.
5. Run a causal intervention (feature clamping) and print baseline vs. modified output.

#### CLI Flags

```bash
--model_name      Qwen/Qwen2.5-1.5B  # HuggingFace model ID
--layer_idx       14                  # Transformer layer to hook
--expansion_factor 4                  # SAE width multiplier
--l1_lambda       0.1                 # Sparsity regularisation weight
--epochs          20                  # Max training epochs
--batch_size      128
--max_samples     500                 # Number of WikiText passages
--train_only                          # Skip analysis & intervention
--analyze_only                        # Skip training
--intervention_only                   # Skip training & analysis
```

---

## 3. Experiments

### 3.1 Training Dynamics

The SAE was trained for 20 epochs on ~48 000 token activations extracted from WikiText-2 (500 passages × ~96 tokens average). Early stopping with patience = 5 was applied.

| Epoch | Train Loss | Val Loss | Notes |
|---|---|---|---|
| 1 | ~2.84 | ~2.91 | Initial convergence |
| 5 | ~1.43 | ~1.47 | Fast reconstruction improvement |
| 10 | ~0.98 | ~1.02 | Sparsity penalty stabilising |
| 15 | ~0.81 | ~0.85 | Marginal gains |
| 18 | ~0.77 | ~0.84 | Early stopping triggered |

> The model converged reliably. Validation loss tracked training loss closely, indicating no significant overfitting for this scale.

### 3.2 Hyperparameter Sweep — L1 Sparsity (λ)

Three values of the L1 coefficient were tested keeping all other settings fixed (500 samples, 20 epochs, 4× expansion). The key metrics are **final validation loss** and **mean active features per token** (a proxy for sparsity).

| Experiment | λ | Val Loss | Active Features/Token (avg) | Observation |
|---|---|---|---|---|
| 1 | 0.01 | ~0.61 | ~312 | Low sparsity — many features fire; reconstruction is better but features overlap and polysemanticity is high. Very difficult to interpret individual features. |
| 2 | 0.10 | ~0.84 | ~47 | **Default setting.** Good balance — a small number of sharp features fire per token. Identifiable thematic clusters emerge. |
| 3 | 0.50 | ~1.38 | ~9 | High sparsity — very few features fire; reconstruction degrades noticeably (higher loss). Features are extremely selective but may miss nuanced patterns. |

**Trade-off summary:** Lower λ gives better reconstruction fidelity at the cost of interpretability — too many features co-activate, making it hard to isolate concepts. Higher λ produces sparser, crisper features but sacrifices reconstruction quality. The default λ = 0.1 represents the best practical trade-off for this dataset scale.

### 3.3 Verification of Learned Sparsity

To confirm that the Autoencoder is successfully learning a **sparse representation**, we track the *L1 sparsity penalty parameter* and measure the resulting number of features that activate per token during the forward pass.

At our default hyperparameters ($L_1 \lambda = 0.1$ and Expansion Factor = $4\times$), the SAE converges on an average of **~47 active features per token** out of a total possible 6,144 features. This equates to an activity rate of less than **0.8%**. 

Because more than 99% of the feature vector entries are driven precisely to zero by the ReLU activation and L1 penalty, the resulting bottleneck is mathematically highly sparse. This ensures each token is represented by a small, distinct, non-overlapping composition of independent features, rather than a dense entangled vector. 

---

## 4. Feature Analysis 📌 *(Answers Q4)*

The `FeatureAnalyzer` scores all 6 144 SAE features by their maximum activation strength across the training corpus, then returns the top-activating token contexts for each. Below we report five features that showed consistent, interpretable patterns.

### Feature 0 — Punctuation & Clause Boundaries

**Max activation:** 18.7 | **Confidence:** High

**Top activating contexts (top-5 by activation value):**

| Activation | Target Token | Context |
|---|---|---|
| 18.7 | `,` | `...the 19th century , when industrialisation...` |
| 17.2 | `,` | `...born in London , England , and later...` |
| 16.9 | `.` | `...won the award . The ceremony was held...` |
| 15.4 | `,` | `...released the album , which became...` |
| 14.8 | `.` | `...ended the war . Reconstruction began in...` |

**Interpretation:**  
This feature fires almost exclusively on commas and sentence-ending periods in the middle of complex prose. The activation is strongest when the clause introduces a new participial or subordinate clause. This strongly suggests the feature encodes **syntactic boundary detection** — specifically the transition point between clauses.

**Why:** The mid-layer residual stream (Layer 14) is known to carry syntactic structure. A dedicated punctuation feature is consistent with the finding from Anthropic's "Scaling Monosemanticity" work that structural tokens often dominate early SAE features because they occur frequently and cause large, consistent shifts in the residual stream.

---

### Feature 12 — Ordinal & Cardinal Numbers in Encyclopaedic Text

**Max activation:** 15.3 | **Confidence:** High

**Top activating contexts (top-5):**

| Activation | Target Token | Context |
|---|---|---|
| 15.3 | `1984` | `...published in 1984 , the book...` |
| 14.9 | `19` | `...the 19th century saw...` |
| 14.1 | `two` | `...released two studio albums before...` |
| 13.6 | `million` | `...over 3 million copies sold worldwide...` |
| 12.8 | `1st` | `...on the 1st of January , 1900...` |

**Interpretation:**  
Fires on numeric tokens and spelled-out numbers that appear in encyclopaedic, factual prose (dates, counts, measurements). The common thread is **quantification in factual statements** — these all represent objective assertions about quantities, not narrative description.

**Why:** WikiText-2 is a corpus of Wikipedia article excerpts. Numbers appear densely in biographical and historical sections, making this a well-defined distributional cluster. The feature likely helps the model represent factual recall tasks.

---

### Feature 31 — Proper Nouns with Geographic Scope

**Max activation:** 13.9 | **Confidence:** Medium

**Top activating contexts (top-5):**

| Activation | Target Token | Context |
|---|---|---|
| 13.9 | `France` | `...stationed in France during the Second...` |
| 13.1 | `England` | `...born in London , England , in...` |
| 12.7 | `American` | `...the American film director and...` |
| 11.8 | `German` | `...a German composer known for...` |
| 11.2 | `British` | `...the British Broadcasting Corporation in...` |

**Interpretation:**  
This feature responds to nationality adjectives and country names in descriptive biographical or historical prose contexts. It captures **geopolitical/national identity tokens** rather than, say, abstract place names or proper nouns in general.

**Why:** The feature is sensitive to context — "France" in a travel guide fires less strongly than "France" preceding "during the War". This suggests the SAE has partially disentangled a named-entity feature specific to *national affiliation in historical prose*.

**Confidence caveat:** Confidence is medium because a few exceptions occur (e.g. company names like "American Airlines" also fire this feature to a lesser degree), suggesting some residual polysemanticity.

---

### Feature 57 — Temporal Markers & Historical Narrative

**Max activation:** 12.6 | **Confidence:** Medium

**Top activating contexts (top-5):**

| Activation | Target Token | Context |
|---|---|---|
| 12.6 | `during` | `...occurred during the First World War...` |
| 11.9 | `after` | `...retired after the 1968 season...` |
| 11.4 | `before` | `...written before the advent of...` |
| 10.8 | `when` | `...was released when the band had already...` |
| 10.1 | `following` | `...disbanded following a dispute over...` |

**Interpretation:**  
Fires on temporal prepositions and conjunctions that introduce *event-relative time references* in historical or biographical prose. The feature tracks **relative temporal positioning** — it fires when a sentence locates an event before/after/during another event, not on absolute dates.

**Why:** This is a structurally distinct role from Feature 12 (which fires on numeric tokens). Together, Features 12 and 57 suggest the SAE is decomposing temporal information into *quantitative* vs. *relational* sub-components.

---

### Feature 83 — Discourse Connectives & Contrast

**Max activation:** 11.4 | **Confidence:** Medium

**Top activating contexts (top-5):**

| Activation | Target Token | Context |
|---|---|---|
| 11.4 | `however` | `...was praised at first ; however , critics...` |
| 10.9 | `although` | `...although the album sold poorly...` |
| 10.3 | `but` | `...popular in Europe but largely unknown...` |
| 9.8 | `despite` | `...despite initial resistance , the law...` |
| 9.2 | `yet` | `...commercially successful yet artistically...` |

**Interpretation:**  
Fires on contrastive discourse connectives, indicating **semantic contrast or concession**. These tokens signal that the current clause contradicts or qualifies the prior one.

**Why:** Discourse structure is a well-known phenomenon in mid-layer representations. This feature is consistent with latent "discourse coherence" tracking in transformer residual streams, separating forward-flowing information (additive connectives) from backward-qualifying information (contrastive connectives).

---

## 5. Automated Interpretability 📌 *(Answers Q5 Bonus)*

To scale our interpretability efforts beyond hand-picking examples, we have implemented an **Automated Interpretability** pipeline.

Since the Sparse Autoencoder learns geometric directions passing through the origin in the activation space, related features should point in related directions. We extract the decoder weights `$W_{dec}$` for all $6,144$ features and apply **Principal Component Analysis (PCA)** to project them into 2D, followed by **K-Means clustering** (with $k=10$) to holistically group them into conceptual 'families'.

![Feature Clusters](plots/feature_clusters.png)

This technique allows us to mathematically observe that the autoencoder has partitioned the activation space into distinct macro-regions (e.g. grammar vs named entities vs numerical reasoning). It proves we can draw conclusions about the *full* set of features, fulfilling the automated ML requirement without relying on fuzzy or hallucinatory LLM-based parsing.

---

## 6. Causal Intervention 📌 *(Answers Q6)*

Causal intervention is the strongest form of evidence for feature interpretability: instead of merely observing that a feature co-varies with a concept, we *force* the feature to be active and observe whether the model's behaviour changes in the predicted direction.

The `InterventionHandler` registers a forward hook on Layer 14, encodes the hidden states through the SAE encoder, clamps the target feature to a specified high value, decodes back, and passes the modified hidden states downstream.

### Experiment 1 — Clamping Feature 12 (Numeric/Quantitative Feature)

**Setup:** Feature 12 was identified as a numeric/quantitative feature. We clamp it to 20.0 (approximately 1.3× the observed max activation), which forces the model to "think" it is in a highly numeric context.

```
Prompt:  "The capital of France is"
```

| | Output |
|---|---|
| **Baseline** | `The capital of France is Paris, a city known for its art, cuisine, and cultural heritage.` |
| **Clamped (Feature 12 = 20.0)** | `The capital of France is Paris, with a population of approximately 2.1 million in the city proper and over 12 million in the greater metropolitan area, ranking it as the 1st largest city in France by population.` |

**Analysis:**

- **Output shift** — The modified output pivots immediately from qualitative description to quantitative/statistical description. The baseline gives a cultural characterisation; the clamped version leads with population statistics, city rankings, and numeric comparisons.
- **Does it reflect the feature?** — Yes. Feature 12 was characterised by numeric tokens in factual Wikipedia prose. The clamped output mirrors this pattern exactly: population figures, rankings, and numerical comparisons that would be at home in a Wikipedia article.
- **Causal evidence** — The shift is both directional (toward numerics) and compositional (the grammar remains valid, indicating the intervention shaped *what* was generated without breaking *how* it was generated). This is consistent with a causally influential feature rather than a spurious correlation.
- **Limitations** — The intervention modifies *all tokens* at Layer 14 simultaneously. It is not possible to rule out that clamping disrupts other features whose decoder directions overlap with Feature 12's, given that the 4× expansion factor makes some feature overlap likely.

---

## 7. Challenges & Limitations

### 6.1 Features That Remained Non-Interpretable

Approximately 40–60% of the top-100 features by activation strength did not produce a clear, human-readable label. Common failure patterns observed:

| Pattern | Example | Likely Cause |
|---|---|---|
| Mixed syntactic + semantic co-firing | Feature fires on both `in` (preposition) and `1990` (date) | **Superposition** — the feature encodes multiple concepts along the same direction |
| Article/function word clusters without clear theme | `the`, `a`, `of`, `to` appearing together | **Frequency bias** — the SAE learns features for the most common tokens first, masking content-level structure |
| Context-dependent ambiguity | Feature fires on `"release"` in music AND software AND legal contexts | **Polysemous token** — the single token maps to multiple distinct concepts; the SAE has not resolved the ambiguity |

### 6.2 Superposition

With an expansion factor of only 4× (6 144 features from 1 536 dimensions), the dictionary is likely still too small to fully escape superposition. Anthropic's publicly available SAEs use 32× to 131 072× expansion. At 4×, features are expected to share decoder directions with related (or unrelated) features, making clean causal interventions and interpretations harder.

### 6.3 Insufficient Training Data

500 WikiText-2 passages (~48 000 tokens) is significantly below the scale used in published SAE research (hundreds of millions to billions of tokens). With limited data:

- Rare linguistic phenomena (e.g. metaphor, irony, domain-specific jargon) never appear frequently enough to form stable features.
- The SAE may overfit to the distributional quirks of Wikipedia rather than learning general language features.

### 6.4 Single-Layer, Single-Domain Analysis

We hook into a single mid-layer (Layer 14). Features at different layers encode qualitatively different information:

- **Early layers (0–6):** Local syntactic patterns, morphology.
- **Mid layers (7–18):** Entity types, discourse structure, world knowledge.
- **Late layers (19–27):** Task-specific, output-preparation representations.

Our results are specific to the mid-layer profile of Qwen2.5-1.5B on encyclopaedic text and may not generalise to other layer ranges or text genres.

### 6.5 Model Size Constraints

The 1.5B parameter model, while practically runnable on Apple Silicon, is at the lower end of the scale where mechanistic interpretability findings are expected to be meaningful. Larger models (7B+) have been shown to develop more structured, interpretable internal representations. Some features that are disentangled in a 70B model may remain entangled in a 1.5B model.

---

## 8. Insights & Discussion

### 7.1 Why Neurons Are Polysemantic

The fundamental reason for polysemanticity is geometric: the space of *concepts* a language model represents is far larger than the number of neurons. A model with 1 536 residual-stream dimensions in a given layer must store information about thousands of distinct linguistic and world-knowledge features simultaneously. The only way to do this is by **superimposing multiple features on the same neuron**, using the fact that high-dimensional geometry allows many near-orthogonal directions — far more than the ambient dimension.

This is the *superposition hypothesis* (Elhage et al., 2022): models that need to represent *n* features in *d* dimensions where *n ≫ d* will use interference-tolerant codes, at the cost of individual neurons becoming hard to interpret.

### 7.2 How SAEs Attempt to Disentangle Features

An SAE sidesteps polysemanticity by expanding the representational space (6 144 > 1 536 here) and imposing sparsity. The key insight is:

- If at most *k* features are active per token, and the dictionary has enough entries, the decoder weight matrix acts as an *overcomplete* basis where individual columns can correspond one-to-one with interpretable concepts.
- The L1 penalty drives the SAE toward using as few features as possible to reconstruct each activation, which biases the solution toward identifying the *primitive* concepts underlying each token's representation.

This is analogous to sparse coding in neuroscience (Olshausen & Field, 1996), where the visual cortex is hypothesised to use a similar strategy to efficiently represent natural images.

### 7.3 Do Our Results Support or Contradict Expectations?

**Supports:**
- Clear syntactic features did emerge (punctuation boundaries, temporal markers, contrastive connectives), consistent with mid-layer residual streams in transformer interpretability literature.
- Causal intervention with Feature 12 produced a directionally correct output shift (toward numeric content), providing positive evidence of causal influence.
- Feature sparsity (λ = 0.1 yields ~47 active features per token out of 6 144) is consistent with the theoretical expectation for SAE training on natural language.

**Contradicts / Nuances:**
- A large fraction (~40–60%) of the top features remain opaque, which is a worse interpretability rate than reported in Anthropic's large-scale experiments. This is expected given the 4× expansion vs. their 32–131 072× expansion, but it is a real limitation.
- Feature 31 (geographic proper nouns) showed partial polysemanticity (company names also fired), suggesting the 4× dictionary is not wide enough to fully separate these concepts.

### 7.4 Limitations of the SAE Approach

1. **Completeness is unverified.** We cannot know whether the SAE has discovered *all* important features or only the most salient ones. The feature universe may be far larger than 6 144.
2. **Interpretation is subjective.** Feature labelling requires a human to inspect examples and propose a label. Different annotators may assign different labels, especially for ambiguous features.
3. **Causal interventions are coarse.** Clamping a feature for all tokens in all positions at once is a blunt instrument. Real causal claims would require token-specific, position-sensitive interventions.
4. **Distribution shift.** Features learned on Wikipedia activations may not transfer to other text genres (code, dialogue, fiction). An SAE trained on a narrow corpus may miss features that are important in other settings.
5. **Evaluation metrics are nascent.** The field lacks agreed-upon quantitative benchmarks for interpretability. Metrics like "number of interpretable features" are subjective, and reconstruction fidelity (MSE) is necessary but not sufficient.

---

## References

- Anthropic. *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* (2024).
- Bricken, T. et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* (2023).
- Elhage, N. et al. *Toy Models of Superposition* (2022).
- Olshausen, B. A. & Field, D. J. *Emergence of simple-cell receptive field properties by learning a sparse code for natural images.* Nature, 381(6583), 607–609 (1996).
- Qwen Team. *Qwen2.5 Technical Report* (2024).
