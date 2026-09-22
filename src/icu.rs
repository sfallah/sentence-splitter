//! The ICU backend: Unicode text segmentation (UAX#29).

use std::sync::LazyLock;

use icu_segmenter::options::SentenceBreakInvariantOptions;
use icu_segmenter::{SentenceSegmenter, SentenceSegmenterBorrowed};

use crate::Newlines;

static SEGMENTER: LazyLock<SentenceSegmenterBorrowed<'static>> =
    LazyLock::new(|| SentenceSegmenter::new(SentenceBreakInvariantOptions::default()));

/// Sentence-end offsets according to UAX#29.
///
/// UAX#29 treats a line break as a paragraph separator, which would end a sentence at
/// every wrapped line. [`Newlines::Space`] avoids that by segmenting a copy in which each
/// `\n` is a space. The copy has the same byte length — one ASCII byte for one ASCII
/// byte — so the offsets it yields are valid for the original text.
pub(crate) fn boundaries(text: &str, newlines: Newlines) -> Vec<usize> {
    let mut ends: Vec<usize> = match newlines {
        Newlines::Separator => SEGMENTER.segment_str(text).collect(),
        Newlines::Space => {
            let flattened: Vec<u8> =
                text.bytes().map(|b| if b == b'\n' { b' ' } else { b }).collect();
            debug_assert_eq!(flattened.len(), text.len());
            SEGMENTER.segment_utf8(&flattened).collect()
        }
    };
    // The iterator starts at 0, which is a sentence *start*, not an end.
    if ends.first() == Some(&0) {
        ends.remove(0);
    }
    ends
}
