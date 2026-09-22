//! Sentence segmentation with two complementary backends.
//!
//! * **Punkt** — Kiss & Strunk's unsupervised algorithm, with data trained per language.
//!   It knows abbreviations, so it keeps `"Dr. Smith went home."` in one piece. It only
//!   works for the 17 languages it has models for, and not at all for scripts without
//!   spaces: Chinese text comes back as a single sentence.
//! * **ICU** — the Unicode text segmentation rules (UAX#29). Script-aware, needs no
//!   training data, and roughly twice as fast, but it has no abbreviation knowledge, so
//!   it splits after `"Dr."`.
//!
//! Neither dominates, which is why both live here behind one API. [`Segmenter::auto`]
//! detects the language and picks: Punkt when a model for that language is available,
//! ICU otherwise.
//!
//! ```
//! use sentence_splitter::{Language, Segmenter};
//!
//! let seg = Segmenter::punkt(Language::English).unwrap();
//! let text = "Dr. Smith went to Washington. He arrived at 5 p.m.";
//! assert_eq!(seg.sentences(text).unwrap().len(), 2);
//! ```
//!
//! The primary API is [`Segmenter::boundaries`], which returns byte offsets and never
//! allocates per sentence; [`Segmenter::spans`] and [`Segmenter::sentences`] build on it,
//! and `sentences` borrows from the input rather than copying it.

#![deny(unsafe_code)]

use std::fmt;

// The vendored Punkt trainer casts `&TrainingData` to `&mut` while training, which is
// unsound but predates this crate. Segmenting with a pretrained model never reaches it.
#[allow(unsafe_code)]
mod punkt;

#[cfg(feature = "detect")]
mod detect;
#[cfg(feature = "icu")]
mod icu;

pub use punkt::params;
pub use punkt::{SentenceByteOffsetTokenizer, SentenceTokenizer, TrainingData};
#[cfg(feature = "train")]
pub use punkt::Trainer;

/// A half-open byte range `[start, end)` into the text that was segmented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// A language with a trained Punkt model.
///
/// Every variant exists regardless of which `lang-*` features are enabled; whether the
/// model is actually compiled in is answered by [`Language::is_available`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Language {
    Czech,
    Danish,
    Dutch,
    English,
    Estonian,
    Finnish,
    French,
    German,
    Greek,
    Italian,
    Norwegian,
    Polish,
    Portuguese,
    Slovene,
    Spanish,
    Swedish,
    Turkish,
}

/// Which algorithm produced (or would produce) a segmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Punkt(Language),
    Icu,
}

/// Whether a line break ends a sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Newlines {
    /// Newlines are treated as spaces, so a sentence may span wrapped lines. The default,
    /// because extracted text (PDF, HTML, scraped pages) wraps mid-sentence constantly.
    ///
    /// Only affects the ICU backend; Punkt already treats a newline as ordinary whitespace.
    Space,
    /// Strict UAX#29: a newline ends a sentence.
    Separator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// No Punkt model for this language is compiled in. Enable its `lang-*` feature.
    ModelUnavailable(Language),
    /// `Segmenter::auto` could not identify the language and the `icu` feature, which it
    /// falls back to, is not enabled.
    NoBackendAvailable,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ModelUnavailable(lang) => write!(
                f,
                "no punkt model for {lang:?}; enable the \"lang-{}\" feature",
                lang.name()
            ),
            Error::NoBackendAvailable => {
                write!(f, "no usable backend: language undetected and the \"icu\" feature is off")
            }
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Punkt(Language),
    #[cfg(feature = "icu")]
    Icu,
    #[cfg(feature = "detect")]
    Auto,
}

/// Splits text into sentences. Cheap to construct and to clone; models are loaded and
/// parsed on first use and then shared for the rest of the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segmenter {
    mode: Mode,
    newlines: Newlines,
}

impl Segmenter {
    /// Punkt with the trained model for `lang`.
    ///
    /// Fails if that model was not compiled in, rather than silently returning no
    /// sentences.
    pub fn punkt(lang: Language) -> Result<Self, Error> {
        if !lang.is_available() {
            return Err(Error::ModelUnavailable(lang));
        }
        Ok(Self { mode: Mode::Punkt(lang), newlines: Newlines::Space })
    }

    /// ICU (UAX#29). Needs no training data and works for any script.
    #[cfg(feature = "icu")]
    pub fn icu() -> Self {
        Self { mode: Mode::Icu, newlines: Newlines::Space }
    }

    /// Detect the language of each text and pick a backend: Punkt when a model for the
    /// detected language is compiled in, ICU otherwise.
    #[cfg(feature = "detect")]
    pub fn auto() -> Self {
        Self { mode: Mode::Auto, newlines: Newlines::Space }
    }

    pub fn with_newlines(mut self, newlines: Newlines) -> Self {
        self.newlines = newlines;
        self
    }

    /// Which backend this segmenter would use for `text`. Constant except for
    /// [`Segmenter::auto`], which decides per text.
    pub fn backend_for(&self, text: &str) -> Result<Backend, Error> {
        match self.mode {
            Mode::Punkt(lang) => Ok(Backend::Punkt(lang)),
            #[cfg(feature = "icu")]
            Mode::Icu => Ok(Backend::Icu),
            #[cfg(feature = "detect")]
            Mode::Auto => {
                match detect::detect(text).filter(|l| l.is_available()) {
                    Some(lang) => Ok(Backend::Punkt(lang)),
                    #[cfg(feature = "icu")]
                    None => Ok(Backend::Icu),
                    #[cfg(not(feature = "icu"))]
                    None => Err(Error::NoBackendAvailable),
                }
            }
        }
    }

    /// Byte offset just past the end of each sentence, ascending. The last element is
    /// always `text.len()`, so the offsets tile the input; empty input gives no offsets.
    ///
    /// This is the primitive the other two methods are built from: it allocates one
    /// `Vec<usize>` and nothing per sentence.
    pub fn boundaries(&self, text: &str) -> Result<Vec<usize>, Error> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut ends = match self.backend_for(text)? {
            Backend::Punkt(lang) => punkt::boundaries(lang, text)?,
            #[cfg(feature = "icu")]
            Backend::Icu => icu::boundaries(text, self.newlines),
            #[cfg(not(feature = "icu"))]
            Backend::Icu => return Err(Error::NoBackendAvailable),
        };
        // Both backends can stop short of the end (Punkt drops trailing whitespace, and a
        // text with no terminator has no final break), so close the last span ourselves.
        match ends.last() {
            Some(&last) if last == text.len() => {}
            _ => ends.push(text.len()),
        }
        Ok(ends)
    }

    /// Sentence spans, contiguous and covering the whole input.
    pub fn spans(&self, text: &str) -> Result<Vec<Span>, Error> {
        let mut start = 0;
        Ok(self
            .boundaries(text)?
            .into_iter()
            .map(|end| {
                let span = Span::new(start, end);
                start = end;
                span
            })
            .collect())
    }

    /// The sentences themselves, borrowed from `text` — concatenating them reproduces the
    /// input exactly.
    pub fn sentences<'t>(&self, text: &'t str) -> Result<Vec<&'t str>, Error> {
        Ok(self.spans(text)?.into_iter().map(|s| &text[s.start..s.end]).collect())
    }
}

impl Language {
    /// The lowercase English name, which is also the `lang-*` feature suffix.
    pub fn name(&self) -> &'static str {
        match self {
            Language::Czech => "czech",
            Language::Danish => "danish",
            Language::Dutch => "dutch",
            Language::English => "english",
            Language::Estonian => "estonian",
            Language::Finnish => "finnish",
            Language::French => "french",
            Language::German => "german",
            Language::Greek => "greek",
            Language::Italian => "italian",
            Language::Norwegian => "norwegian",
            Language::Polish => "polish",
            Language::Portuguese => "portuguese",
            Language::Slovene => "slovene",
            Language::Spanish => "spanish",
            Language::Swedish => "swedish",
            Language::Turkish => "turkish",
        }
    }

    /// Whether this language's Punkt model was compiled in.
    pub fn is_available(&self) -> bool {
        punkt::training_data(*self).is_some()
    }

    /// Every language whose model is compiled in.
    pub fn available() -> Vec<Language> {
        const ALL: [Language; 17] = [
            Language::Czech,
            Language::Danish,
            Language::Dutch,
            Language::English,
            Language::Estonian,
            Language::Finnish,
            Language::French,
            Language::German,
            Language::Greek,
            Language::Italian,
            Language::Norwegian,
            Language::Polish,
            Language::Portuguese,
            Language::Slovene,
            Language::Spanish,
            Language::Swedish,
            Language::Turkish,
        ];
        ALL.into_iter().filter(Language::is_available).collect()
    }
}
