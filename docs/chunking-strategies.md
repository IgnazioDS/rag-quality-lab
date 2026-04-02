# Chunking Strategies

This document explains the three chunking strategies implemented in this lab: how each
algorithm works, what corpus characteristics it handles well and poorly, which parameters
to tune first, and a worked example showing how each strategy processes the same passage.

---

## 1. Fixed-Size (`fixed_size`)

### Algorithm

1. Walk the text in steps of `chunk_size - overlap` characters.
2. At each step, slice `text[start : start + chunk_size]`.
3. Record `char_start`, `char_end`, and token count (via tiktoken `cl100k_base`).
4. Advance `start` by `step`. The last chunk may be shorter than `chunk_size`.

The overlap ensures that information near chunk boundaries appears in two consecutive
chunks. This reduces the chance that a query spanning a boundary finds nothing.

**Key parameters:**
- `chunk_size` (default: 512): Maximum chunk length in characters.
- `overlap` (default: 64): Number of characters repeated at the start of the next chunk.

**Tune `chunk_size` first.** Smaller values increase recall by producing more specific
chunks, but increase embedding cost and storage. Larger values improve context for
multi-sentence reasoning but reduce precision. Start at 400–600 characters.

**Tune `overlap` second.** 10–15% of `chunk_size` is a reasonable default. Higher
overlap costs more storage and retrieval latency without proportional quality gain.

### Handles well
- Homogeneous text (e.g., a uniform prose document with no section structure)
- Baseline comparison — easiest to reason about, minimal assumptions
- Corpora where sentence/paragraph boundaries have no semantic significance

### Handles poorly
- Documents with strong section boundaries: chunks split mid-sentence or mid-paragraph
- Code: a function split across two chunks is likely useless in either
- Tabular data: rows may be split arbitrarily

---

## 2. Recursive (`recursive`)

### Algorithm

1. Try to split the text with the first separator (`"\n\n"`).
2. If any resulting piece exceeds `max_tokens`, recurse into that piece using the next
   separator (`"\n"`, then `". "`, then `" "`, then character-by-character).
3. Stop recursing when all pieces are within `max_tokens` or no more separators remain.
4. Token count (not character count) is the authoritative size constraint.

The separator hierarchy ensures that natural structure is preserved: paragraph breaks
are respected before falling back to sentence boundaries, which are respected before
word boundaries.

**Key parameters:**
- `max_tokens` (default: 400): Maximum tokens per chunk.

**Tune `max_tokens` first.** 300–500 is appropriate for most retrieval use cases.
Smaller values produce more precise chunks that better match short queries. Larger
values preserve more context per chunk.

### Handles well
- Structured documents: articles, documentation, README files, technical writing
- Text where paragraphs are semantically coherent units
- Mixed-length content: short paragraphs stay intact, long ones are further split

### Handles poorly
- Unstructured text with no separator cues (rare in real corpora)
- Transcripts or conversational text with irregular paragraph usage
- Documents where `"\n\n"` is used for visual formatting rather than semantic structure

---

## 3. Semantic (`semantic`)

### Algorithm

1. Split the text into sentences using boundary detection (`[.!?]` followed by whitespace).
2. Embed all sentences in a **single batched API call**.
3. Walk the sentence list. For each new sentence:
   a. Compute cosine similarity between the sentence embedding and the **centroid** of
      the current chunk's sentence embeddings.
   b. If similarity < `similarity_threshold` OR adding this sentence would exceed
      `max_tokens`: flush the current chunk and start a new one.
4. The chunk centroid is the mean of all sentence embeddings in the current group,
   recomputed after each addition.

The batch embedding call is critical for performance — embedding N sentences with N
API calls would make this strategy prohibitively slow for large corpora.

**Key parameters:**
- `similarity_threshold` (default: 0.8): Cosine similarity below which a new chunk begins.
- `max_tokens` (default: 400): Hard token limit regardless of similarity.

**Tune `similarity_threshold` first.** Lower values (0.6–0.7) produce larger, more
contextually mixed chunks. Higher values (0.85–0.95) produce more focused chunks that
may be too short to contain enough context for retrieval. 0.75–0.85 is the productive
range for most English prose.

### Handles well
- Documents where topic shifts do not align with paragraph breaks
- Narrative text and long-form content
- Corpora where adjacent sentences differ widely in topic despite being in the same paragraph

### Handles poorly
- Very short documents (fewer than 5 sentences) — the batched embedding call overhead
  may not amortize
- Highly technical or domain-specific text where the embedding model may not produce
  meaningful similarity signals
- Documents with many single-sentence paragraphs (devolves to sentence-level chunking)

---

## Worked Example: Same Passage, Three Strategies

**Input passage (approx. 500 characters / 85 tokens):**

> The city of Florence was founded by Julius Caesar in 59 BC as a settlement for his
> veterans. Over the following centuries it grew to become a major centre of medieval
> European trade and finance. The Renaissance began there in the 14th century, driven
> by wealthy merchant families like the Medici, who became powerful patrons of art and
> architecture. Today Florence is best known for its museums and galleries, particularly
> the Uffizi, which holds one of the world's greatest collections of Italian Renaissance art.

---

### fixed_size (chunk_size=200, overlap=40)

**Chunk 0** (chars 0–200):
> The city of Florence was founded by Julius Caesar in 59 BC as a settlement for his
> veterans. Over the following centuries it grew to become a major centre of medieval
> European trade and fin

**Chunk 1** (chars 160–360):
> European trade and finance. The Renaissance began there in the 14th century, driven
> by wealthy merchant families like the Medici, who became powerful patrons of art and
> archite

**Chunk 2** (chars 320–500):
> rchitecture. Today Florence is best known for its museums and galleries, particularly
> the Uffizi, which holds one of the world's greatest collections of Italian Renaissance art.

**Observation:** Chunks split mid-word ("fin-ance", "archite-cture"). The overlap
helps continuity but produces redundant content and broken tokens at boundaries.

---

### recursive (max_tokens=30)

**Chunk 0**:
> The city of Florence was founded by Julius Caesar in 59 BC as a settlement for his veterans.

**Chunk 1**:
> Over the following centuries it grew to become a major centre of medieval European trade and finance.

**Chunk 2**:
> The Renaissance began there in the 14th century, driven by wealthy merchant families like the Medici, who became powerful patrons of art and architecture.

**Chunk 3**:
> Today Florence is best known for its museums and galleries, particularly the Uffizi, which holds one of the world's greatest collections of Italian Renaissance art.

**Observation:** Splits cleanly at sentence boundaries. Each chunk is a complete
sentence with coherent meaning. No mid-word breaks.

---

### semantic (similarity_threshold=0.8, max_tokens=60)

**Chunk 0** (sentences 0–1 are topically similar — Roman history and medieval trade):
> The city of Florence was founded by Julius Caesar in 59 BC as a settlement for his veterans. Over the following centuries it grew to become a major centre of medieval European trade and finance.

**Chunk 1** (sentences 2–3 shift to Renaissance and arts):
> The Renaissance began there in the 14th century, driven by wealthy merchant families like the Medici, who became powerful patrons of art and architecture. Today Florence is best known for its museums and galleries, particularly the Uffizi, which holds one of the world's greatest collections of Italian Renaissance art.

**Observation:** The split occurs at the semantic boundary between "medieval trade"
and "Renaissance arts" — a topic shift that the recursive strategy, constrained only
by token count, happens to also capture here but would miss on longer inputs where
multiple sentences describe the same topic before shifting.
