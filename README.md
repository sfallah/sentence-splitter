# sentence-splitter

Sentence segmentation with two complementary backends behind one API.

| | Punkt | ICU |
|---|---|---|
| approach | unsupervised, trained per language | Unicode rules (UAX#29) |
| knows abbreviations | yes — keeps `"Dr. Smith went home."` whole | no — splits after `"Dr."` |
| scripts without spaces | no — Chinese comes back as one sentence | yes |
| languages | 17 with trained models | any |
| speed | 80 MiB/s | **234 MiB/s** |

Neither wins outright, which is why both are here. `Segmenter::auto()` detects the
language and picks: Punkt when a model for it is compiled in, ICU otherwise — so Chinese
and Japanese get sensible results instead of one giant sentence.

```rust
use sentence_splitter::{Language, Segmenter};

// Pick a backend explicitly...
let seg = Segmenter::punkt(Language::English)?;
assert_eq!(
    seg.sentences("Dr. Smith went to Washington. He arrived at 5 p.m.")?,
    ["Dr. Smith went to Washington.", " He arrived at 5 p.m."]
);

// ...or let it choose per text.
let auto = Segmenter::auto();
assert_eq!(auto.sentences("这是第一句。这是第二句。")?.len(), 2); // falls back to ICU
```

Three views of the same segmentation, cheapest first:

```rust
seg.boundaries(text)?; // Vec<usize> — byte offset of each sentence end
seg.spans(text)?;      // Vec<Span>  — contiguous [start, end) ranges
seg.sentences(text)?;  // Vec<&str>  — borrowed from `text`, not copied
```

All three tile the input: the sentences concatenated are the input, byte for byte.

## Languages and binary size

Each language's trained model is embedded at compile time and **parsed on first use**, so
enabling several costs binary size, not startup time. Models are 0.3 MB (English) to
1.7 MB (Greek); all 17 together are 15 MB. Only English is on by default:

```toml
sentence-splitter = "0.1"                                             # English + ICU + detection
sentence-splitter = { version = "0.1", features = ["lang-german"] }   # add one
sentence-splitter = { version = "0.1", features = ["lang-all"] }      # all 17 (15 MB)
sentence-splitter = { version = "0.1", default-features = false, features = ["icu"] }  # ICU only
```

Asking for a language whose feature is off returns `Error::ModelUnavailable` rather than
silently yielding no sentences.

| feature | default | what it adds |
|---|:-:|---|
| `icu` | ✓ | the ICU backend |
| `detect` | ✓ | `Segmenter::auto`, via `whatlang` |
| `lang-english` | ✓ | the English Punkt model |
| `lang-*` (16 more), `lang-all` | | the other trained models |
| `train` | | `Trainer`, for building new models — see the caveat below |

## Newlines

Extracted text (PDF, HTML, scraped pages) wraps mid-sentence, and UAX#29 treats a line
break as a paragraph separator. `Newlines::Space` — the default — segments as if newlines
were spaces, so a wrapped sentence stays whole. `Newlines::Separator` gives strict UAX#29.
The setting only affects ICU; Punkt already treats newlines as ordinary whitespace.

## Attribution and licence

The Punkt implementation in `src/punkt/` is vendored from
[ferristseng/rust-punkt](https://github.com/ferristseng/rust-punkt) (MIT OR Apache-2.0),
which implements Kiss & Strunk's algorithm. Changes: ported to edition 2021, JSON parsing
moved off the unmaintained `rustc-serialize` to `serde_json`, per-language features, and a
byte-offset API that does not allocate per sentence.

This crate is likewise MIT OR Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.

**Caveat on `train`:** the vendored trainer casts `&TrainingData` and `&Token` to `&mut`
to mutate them while an iterator still borrows them — undefined behaviour, which rustc's
`invalid_reference_casting` lint rejects (cargo only hides it upstream because it caps
lints in dependencies). Fixing it means preserving the mutate-while-iterating semantics
the algorithm depends on, so rather than change training results by guesswork it is left
as-is behind an off-by-default feature. **Using the pretrained models never reaches this
code**, and the default build does not compile it.
