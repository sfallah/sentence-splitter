//! Language detection, used by [`crate::Segmenter::auto`].

use whatlang::Lang;

use crate::Language;

/// Detect the language of `text`, or `None` if it is not one Punkt has a model for.
///
/// Only the first 1024 characters are examined: detection converges quickly and this
/// keeps the cost flat for long documents.
pub(crate) fn detect(text: &str) -> Option<Language> {
    let sample: String = text.chars().take(1024).collect();
    from_whatlang(whatlang::detect(&sample)?.lang())
}

fn from_whatlang(lang: Lang) -> Option<Language> {
    Some(match lang {
        Lang::Ces => Language::Czech,
        Lang::Dan => Language::Danish,
        Lang::Nld => Language::Dutch,
        Lang::Eng => Language::English,
        Lang::Est => Language::Estonian,
        Lang::Fin => Language::Finnish,
        Lang::Fra => Language::French,
        Lang::Deu => Language::German,
        Lang::Ell => Language::Greek,
        Lang::Ita => Language::Italian,
        Lang::Nob => Language::Norwegian,
        Lang::Pol => Language::Polish,
        Lang::Por => Language::Portuguese,
        Lang::Slv => Language::Slovene,
        Lang::Spa => Language::Spanish,
        Lang::Swe => Language::Swedish,
        Lang::Tur => Language::Turkish,
        _ => return None,
    })
}
