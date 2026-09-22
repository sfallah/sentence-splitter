use sentence_splitter::{Backend, Language, Newlines, Segmenter};

/// Every backend must tile its input: the sentences, concatenated, are the input.
fn assert_tiles(seg: &Segmenter, text: &str) {
    let sentences = seg.sentences(text).unwrap();
    assert_eq!(sentences.concat(), text, "sentences do not reconstruct {text:?}");
    let spans = seg.spans(text).unwrap();
    let mut cursor = 0;
    for span in &spans {
        assert_eq!(span.start, cursor, "gap or overlap at {span:?} in {text:?}");
        cursor = span.end;
    }
    assert_eq!(cursor, text.len(), "spans stop short in {text:?}");
}

fn every_segmenter() -> Vec<Segmenter> {
    vec![Segmenter::punkt(Language::English).unwrap(), Segmenter::icu(), Segmenter::auto()]
}

const TEXTS: [&str; 7] = [
    "Dr. Smith went to Washington. He arrived at 5 p.m. yesterday.",
    "One sentence with no terminator",
    "Trailing whitespace after the end.   ",
    "\n\nLeading blank lines. Then text.\n",
    "Wrapped text does\nnot end a sentence. The next one starts here.",
    "这是第一句。这是第二句。",
    "",
];

#[test]
fn every_backend_tiles_its_input() {
    for seg in every_segmenter() {
        for text in TEXTS {
            assert_tiles(&seg, text);
        }
    }
}

#[test]
fn punkt_keeps_abbreviations_together() {
    let seg = Segmenter::punkt(Language::English).unwrap();
    let text = "Dr. Smith went to Washington. He arrived at 5 p.m. yesterday.";
    assert_eq!(
        seg.sentences(text).unwrap(),
        ["Dr. Smith went to Washington.", " He arrived at 5 p.m. yesterday."]
    );
}

#[test]
fn icu_segments_scripts_without_spaces() {
    // The case punkt cannot do: it returns Chinese text as a single sentence.
    let text = "这是第一句。这是第二句。";
    assert_eq!(Segmenter::icu().sentences(text).unwrap().len(), 2);
    assert_eq!(Segmenter::punkt(Language::English).unwrap().sentences(text).unwrap().len(), 1);
}

#[test]
fn newline_policy_only_affects_icu() {
    let text = "A wrapped line\ncontinues here. A second sentence.";
    let spans = |n| Segmenter::icu().with_newlines(n).sentences(text).unwrap().len();
    assert_eq!(spans(Newlines::Space), 2, "newlines as spaces must not end a sentence");
    assert_eq!(spans(Newlines::Separator), 3, "strict UAX#29 ends a sentence at the newline");

    let punkt = |n| Segmenter::punkt(Language::English).unwrap().with_newlines(n).sentences(text).unwrap();
    assert_eq!(punkt(Newlines::Space), punkt(Newlines::Separator));
}

#[test]
fn auto_picks_punkt_for_known_languages_and_icu_otherwise() {
    let auto = Segmenter::auto();
    let english = "This is a reasonably long English text, long enough for the detector \
                   to identify the language with confidence. It has several sentences.";
    assert_eq!(auto.backend_for(english).unwrap(), Backend::Punkt(Language::English));
    // Chinese has no punkt model compiled in, so auto falls back to ICU.
    assert_eq!(auto.backend_for("这是第一句。这是第二句。").unwrap(), Backend::Icu);
}

#[test]
fn missing_model_is_an_error_not_an_empty_result() {
    // German's model is not in the default feature set.
    match Segmenter::punkt(Language::German) {
        Err(e) => assert!(e.to_string().contains("lang-german"), "{e}"),
        Ok(_) => assert!(Language::German.is_available()),
    }
}

#[test]
fn available_languages_match_enabled_features() {
    let available = Language::available();
    assert!(available.contains(&Language::English), "default features include lang-english");
    assert!(available.iter().all(Language::is_available));
}

#[test]
fn boundaries_are_ascending_and_end_at_the_input_length() {
    for seg in every_segmenter() {
        for text in TEXTS.iter().filter(|t| !t.is_empty()) {
            let ends = seg.boundaries(text).unwrap();
            assert_eq!(*ends.last().unwrap(), text.len(), "{text:?}");
            assert!(ends.windows(2).all(|w| w[0] < w[1]), "not ascending: {ends:?} for {text:?}");
            assert!(ends.iter().all(|&e| text.is_char_boundary(e)), "split a character in {text:?}");
        }
        assert!(seg.boundaries("").unwrap().is_empty());
    }
}
