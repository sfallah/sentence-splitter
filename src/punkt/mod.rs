//! The Punkt implementation, vendored from `ferristseng/rust-punkt` (MIT OR Apache-2.0).
//!
//! Changes from upstream: ported to edition 2021, JSON parsing moved from the
//! unmaintained `rustc-serialize` to `serde_json`, and each language's trained data is
//! behind its own `lang-*` feature so only the models you ask for are compiled in.

// A good deal of the vendored implementation (the collocation types, the log-likelihood
// scorers, the orthography tables) exists only to serve the `train` feature, so it reads
// as dead code whenever that feature is off.
#![allow(dead_code)]

pub(crate) mod freqdist;
pub(crate) mod prelude;
pub(crate) mod token;
pub(crate) mod tokenizer;
pub(crate) mod trainer;
#[cfg(feature = "train")]
pub(crate) mod train;
pub(crate) mod util;

use std::sync::LazyLock;

use crate::{Error, Language};

pub use tokenizer::{SentenceByteOffsetTokenizer, SentenceTokenizer};
pub use trainer::TrainingData;
#[cfg(feature = "train")]
pub use train::Trainer;

/// Traits for configuring the tokenizers and the trainer, plus the default parameters.
pub mod params {
    pub use super::prelude::{
        DefinesInternalPunctuation, DefinesNonPrefixCharacters, DefinesNonWordCharacters,
        DefinesPunctuation, DefinesSentenceEndings, Set, Standard, TrainerParameters,
    };
}

// One lazily-parsed model per enabled language. The JSON is embedded at compile time but
// only parsed the first time that language is actually segmented, so enabling several
// languages costs binary size, not startup time.
macro_rules! lazy_models {
    ($( $feature:literal => ($variant:ident, $loader:ident, $static_name:ident) ),* $(,)?) => {
        $(
            #[cfg(feature = $feature)]
            static $static_name: LazyLock<TrainingData> = LazyLock::new(TrainingData::$loader);
        )*

        /// The trained model for `lang`, or `None` if its feature is disabled.
        pub(crate) fn training_data(lang: Language) -> Option<&'static TrainingData> {
            match lang {
                $(
                    #[cfg(feature = $feature)]
                    Language::$variant => Some(&$static_name),
                )*
                #[allow(unreachable_patterns)]
                _ => None,
            }
        }
    };
}

lazy_models! {
    "lang-czech" => (Czech, czech, CZECH),
    "lang-danish" => (Danish, danish, DANISH),
    "lang-dutch" => (Dutch, dutch, DUTCH),
    "lang-english" => (English, english, ENGLISH),
    "lang-estonian" => (Estonian, estonian, ESTONIAN),
    "lang-finnish" => (Finnish, finnish, FINNISH),
    "lang-french" => (French, french, FRENCH),
    "lang-german" => (German, german, GERMAN),
    "lang-greek" => (Greek, greek, GREEK),
    "lang-italian" => (Italian, italian, ITALIAN),
    "lang-norwegian" => (Norwegian, norwegian, NORWEGIAN),
    "lang-polish" => (Polish, polish, POLISH),
    "lang-portuguese" => (Portuguese, portuguese, PORTUGUESE),
    "lang-slovene" => (Slovene, slovene, SLOVENE),
    "lang-spanish" => (Spanish, spanish, SPANISH),
    "lang-swedish" => (Swedish, swedish, SWEDISH),
    "lang-turkish" => (Turkish, turkish, TURKISH),
}

/// Sentence-end offsets according to Punkt. Uses the byte-offset tokenizer, so no text is
/// copied.
pub(crate) fn boundaries(lang: Language, text: &str) -> Result<Vec<usize>, Error> {
    let data = training_data(lang).ok_or(Error::ModelUnavailable(lang))?;
    Ok(
        SentenceByteOffsetTokenizer::<params::Standard>::new(text, data)
            .map(|(_start, end)| end)
            .collect(),
    )
}
