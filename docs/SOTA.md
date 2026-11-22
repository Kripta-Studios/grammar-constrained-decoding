# State of the art: grammar-constrained decoding (late 2026)

Only sources I actually looked at are listed. Numbers are the authors' claims, not
measurements from this repo.

## Core papers

| Work | Year | Idea | Link |
|---|---|---|---|
| **Efficient Guided Generation for LLMs** (Willard & Louf), the paper behind *Outlines* | 2023 | Compile a regex to an FSM, then precompute for every FSM state which vocabulary tokens keep the FSM alive (an *index*). Each step is then a lookup. Extends to CFGs by combining the FSM index with an LALR(1) parser. | [arXiv:2307.09702](https://arxiv.org/abs/2307.09702) |
| **XGrammar** (Dong et al.) | 2024 | A byte-level pushdown automaton with a **persistent stack** for cheap branching and rollback. Splits the vocabulary at each automaton position into *context-independent* tokens (decided from the position alone, precomputed in an **adaptive token mask cache** stored as an accept list, a reject list or a bitset, whichever is smallest) and *context-dependent* tokens (checked at runtime with the full stack). Adds context expansion to shrink the context-dependent set, and overlaps mask computation with the GPU forward pass. Claims >99% of a mask can be precomputed and up to 100x lower per-token latency than earlier CFG engines. | [arXiv:2411.15100](https://arxiv.org/abs/2411.15100), [project page](https://catalyst.cs.cmu.edu/projects/xgrammar.html) |
| **XGrammar-2** | 2026 | Targets agent workloads where the schema changes within and across requests. Adds *TagDispatch* (switch grammars on trigger tags), a cross-grammar cache, an **Earley-based** adaptive token mask cache, JIT compilation and repetition-state compression. Claims >6x faster compilation and near-zero end-to-end overhead in serving engines. | [arXiv:2601.04426](https://arxiv.org/abs/2601.04426) |
| **JSONSchemaBench** (guidance-ai et al.) | 2025 | Benchmark of about 10K real-world JSON Schemas. Compares Guidance, Outlines, llama.cpp, XGrammar, OpenAI and Gemini on efficiency, schema-feature coverage and output quality. Finds constrained decoding can *speed up* generation (forced tokens) and slightly improve downstream accuracy. The main point for us: **coverage** of JSON Schema features is where engines differ most. | [arXiv:2501.10868](https://arxiv.org/abs/2501.10868), [code](https://github.com/guidance-ai/jsonschemabench) |
| Accelerating Constrained Decoding with Token Space Compression | 2026 | Notes that Earley parsing (used under XGrammar-2) is linear-time for some grammar classes but O(n^3) in the worst case, and compresses the token space to cut mask cost. | [arXiv:2605.29986](https://arxiv.org/abs/2605.29986) |

## Production engines

* **llguidance** (guidance-ai, Rust). An Earley parser for the CFG on top of a lexer
  built from **regex derivatives**. The mask is computed *on the fly* (no big
  precomputation) by walking the token trie. Most trie nodes are handled by the lexer;
  the Earley parser only runs at lexeme boundaries. The README reports per-mask times of
  tens of microseconds for a 128k vocabulary on JSON-schema grammars. Its benchmark is
  called MaskBench. [repo](https://github.com/guidance-ai/llguidance),
  [blog: "making structured outputs go brrr"](https://guidance-ai.github.io/blog/llguidance-making-structured-outputs-go-brrr)
* **Outlines** (dottxt). The FSM/index approach above. Very fast at decode time for
  regular languages (JSON Schema without recursion compiles to a regex). Compiling the
  index can be expensive for large schemas.
* **vLLM**. "Structured outputs" with pluggable backends (xgrammar, guidance/llguidance,
  outlines). The bitmask is applied to the logits on the GPU, and grammar state advances
  per request in the scheduler. [vLLM blog: structured decoding intro](https://vllm.ai/blog/struct-decode-intro),
  [xgrammar backend docs](https://docs.vllm.ai/en/latest/api/vllm/v1/structured_output/backend_xgrammar/),
  [Red Hat: structured outputs in vLLM](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses)
* **SGLang**. Integrates XGrammar (the XGrammar-2 paper says it has been added to both SGLang and vLLM).

## How the main ideas relate (and where this repo sits)

| | Outlines (FSM) | XGrammar | llguidance | **this repo** |
|---|---|---|---|---|
| Language class | regular (CFG via LALR) | CFG (PDA); Earley in v2 | CFG (Earley + lexer) | CFG, nondeterministic byte PDA |
| Granularity | characters/bytes | bytes | lexemes + bytes | bytes |
| Mask strategy | full precomputed index per FSM state | precomputed CI cache + runtime CD check | on-the-fly trie walk | lazy CI cache keyed by stack suffix + runtime CD check |
| Vocabulary walk | n/a (index) | sorted vocabulary + prefix sharing | trie | sorted vocabulary + LCP array (implicit trie) |
| Implementation | Python + Rust core | C++ | Rust | pure Python (didactic) |

Recurring lessons from these papers:

1. **Work at the byte level.** BPE tokens do not align with grammar terminals or even
   with UTF-8 characters.
2. **Most of the mask does not depend on the stack below the top.** Caching that part is
   the main speedup.
3. **Overlap the CPU mask with the GPU forward pass.** Kernel launches are asynchronous,
   so the mask can be computed while the GPU works.
4. **Coverage and semantics matter more than raw speed** once the mask takes under a
   millisecond (JSONSchemaBench).
