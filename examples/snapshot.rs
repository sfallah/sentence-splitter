//! Fingerprints detection and segmentation output, so a dependency bump can be shown to
//! change nothing.
//!
//! The integration tests assert structural properties — sentences tile the input, offsets
//! ascend, markers are recognised — which hold whatever the detector and the trained models
//! do. They would stay green if `whatlang` started classifying a text differently and
//! `Segmenter::auto` silently rerouted it from Punkt to ICU, or if a model's abbreviation
//! set shifted. This prints the actual decisions and boundaries instead:
//!
//! ```text
//! cargo run --release --all-features --example snapshot > /tmp/before.txt
//! # ...edit Cargo.toml, cargo update -p <crate>...
//! cargo run --release --all-features --example snapshot > /tmp/after.txt
//! diff /tmp/before.txt /tmp/after.txt
//! ```
//!
//! Run with `--all-features` so every language model is present; otherwise the detected
//! language often has no model and everything falls back to ICU, hiding the interesting part.

use sentence_splitter::{Backend, Language, Newlines, Segmenter, TrainingData};

/// Samples long enough for the detector to be confident. The last three have no Punkt
/// model, so `auto` must fall back to ICU.
const SAMPLES: &[(&str, &str)] = &[
    ("en", "The committee met on Tuesday to review the proposal. Dr. Smith presented the findings at 4 p.m. Several members raised concerns about the budget."),
    ("de", "Der Ausschuss traf sich am Dienstag, um den Vorschlag zu prüfen. Die Kosten belaufen sich auf 5 Mio. Euro. Mehrere Mitglieder äußerten Bedenken."),
    ("fr", "Le comité s'est réuni mardi pour examiner la proposition. Le docteur Dupont a présenté les résultats. Plusieurs membres ont exprimé des inquiétudes."),
    ("es", "El comité se reunió el martes para revisar la propuesta. El doctor Pérez presentó los resultados. Varios miembros expresaron su preocupación."),
    ("it", "Il comitato si è riunito martedì per esaminare la proposta. Il dottor Rossi ha presentato i risultati. Diversi membri hanno espresso preoccupazioni."),
    ("pt", "O comitê reuniu-se na terça-feira para analisar a proposta. O doutor Silva apresentou os resultados. Vários membros manifestaram preocupações."),
    ("nl", "De commissie kwam dinsdag bijeen om het voorstel te bespreken. Dokter Jansen presenteerde de bevindingen. Verschillende leden uitten hun zorgen."),
    ("tr", "Komite teklifi görüşmek üzere salı günü toplandı. Doktor Yılmaz bulguları sundu. Birkaç üye bütçe konusunda endişelerini dile getirdi."),
    ("el", "Η επιτροπή συνεδρίασε την Τρίτη για να εξετάσει την πρόταση. Ο γιατρός παρουσίασε τα ευρήματα. Αρκετά μέλη εξέφρασαν ανησυχίες."),
    ("cs", "Výbor se sešel v úterý, aby projednal návrh. Doktor Novák představil zjištění. Několik členů vyjádřilo obavy ohledně rozpočtu."),
    ("zh", "委员会星期二开会审议该提案。史密斯医生介绍了调查结果。几位成员对预算表示担忧。"),
    ("ja", "委員会は火曜日に提案を検討するために開催されました。スミス医師が調査結果を発表しました。数名の委員が予算について懸念を示しました。"),
    ("ru", "Комитет собрался во вторник для рассмотрения предложения. Доктор Смирнов представил результаты. Несколько членов выразили обеспокоенность."),
];

fn main() {
    // 1. Which models are compiled in — the feature wiring.
    let available = Language::available();
    println!(
        "models    n={:<3} {}",
        available.len(),
        available
            .iter()
            .map(|l| l.name())
            .collect::<Vec<_>>()
            .join(",")
    );

    // 2. Trained-data lookups. Exercises the JSON parse and the hash/phf containers
    //    that hold abbreviations, sentence starters and collocations.
    let probes: &[(&str, &str)] = &[("va", "among"), ("w.va", "the"), ("dr", "however")];
    let eng = TrainingData::english();
    for (abbrev, starter) in probes {
        println!(
            "traindata abbrev({abbrev})={:<5} starter({starter})={:<5} ortho({starter})={}",
            eng.contains_abbrev(abbrev),
            eng.contains_sentence_starter(starter),
            eng.get_orthographic_context(starter)
        );
    }
    println!(
        "traindata collocation(##number##,corrections)={}",
        eng.contains_collocation("##number##", "corrections")
    );

    // 3. What the detector decides, and therefore which backend `auto` picks.
    //    This is the part `whatlang` controls.
    let auto = Segmenter::auto();
    for (tag, text) in SAMPLES {
        let backend = auto.backend_for(text).unwrap();
        let picked = match backend {
            Backend::Punkt(l) => l.name(),
            Backend::Icu => "icu",
        };
        println!(
            "detect    {tag}  -> {:<12} boundaries={:?}",
            picked,
            auto.boundaries(text).unwrap()
        );
    }

    // 4. Each backend explicitly, so a change in either is attributable.
    for (tag, text) in SAMPLES {
        let icu = Segmenter::icu();
        println!(
            "icu       {tag}  sep={:?} space={:?}",
            icu.with_newlines(Newlines::Separator)
                .boundaries(text)
                .unwrap(),
            icu.with_newlines(Newlines::Space).boundaries(text).unwrap()
        );
    }
    for lang in &available {
        let seg = Segmenter::punkt(*lang).unwrap();
        // Same text through every model: differences are the models, not the input.
        let text = SAMPLES[0].1;
        println!(
            "punkt     {:<11} boundaries={:?}",
            lang.name(),
            seg.boundaries(text).unwrap()
        );
    }

    // 5. The awkward inputs the tests cover, through every backend.
    let edge: &[&str] = &[
        "",
        "no terminator",
        "trailing space.   ",
        "\n\nblank lines. then text.\n",
        "wrapped text does\nnot end a sentence. next one here.",
    ];
    for (i, text) in edge.iter().enumerate() {
        println!(
            "edge      {i} punkt={:?} icu={:?} auto={:?}",
            Segmenter::punkt(Language::English)
                .unwrap()
                .boundaries(text)
                .unwrap(),
            Segmenter::icu().boundaries(text).unwrap(),
            auto.boundaries(text).unwrap()
        );
    }
}
