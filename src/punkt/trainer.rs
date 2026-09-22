// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::default::Default;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::str::FromStr;
use hashbrown::{HashMap, HashSet};
use serde_json::Value;

use crate::punkt::prelude::OrthographicContext;
use crate::punkt::token::Token;

/// A collocation is any pair of words that has a high likelihood of appearing
/// together.
#[derive(Debug, Eq)]
pub struct Collocation<T>
where
  T: Deref<Target = Token>,
{
  l: T,
  r: T,
}

impl<T> Collocation<T>
where
  T: Deref<Target = Token>,
{
  #[inline(always)]
  pub fn new(l: T, r: T) -> Collocation<T> {
    Collocation { l: l, r: r }
  }

  #[inline(always)]
  pub fn left(&self) -> &T {
    &self.l
  }

  #[inline(always)]
  pub fn right(&self) -> &T {
    &self.r
  }
}

impl<T> Hash for Collocation<T>
where
  T: Deref<Target = Token>,
{
  #[inline(always)]
  fn hash<H>(&self, state: &mut H)
  where
    H: Hasher,
  {
    (*self.l).typ_without_period().hash(state);
    (*self.r).typ_without_break_or_period().hash(state);
  }
}

impl<T> PartialEq for Collocation<T>
where
  T: Deref<Target = Token>,
{
  #[inline(always)]
  fn eq(&self, x: &Collocation<T>) -> bool {
    (*self.l).typ_without_period() == (*x.l).typ_without_period()
      && (*self.r).typ_without_break_or_period() == (*x.r).typ_without_break_or_period()
  }
}

/// Stores data that was obtained during training.
///
/// # Examples
///
/// Precompiled data can be loaded via a language specific constructor.
///
/// ```
/// # use sentence_splitter::TrainingData;
/// #
/// // One constructor per enabled `lang-*` feature.
/// let eng_data = TrainingData::english();
///
/// assert!(eng_data.contains_abbrev("va"));
/// ```
#[derive(Debug, Default)]
pub struct TrainingData {
  abbrevs: HashSet<String>,
  collocations: HashMap<String, HashSet<String>>,
  sentence_starters: HashSet<String>,
  orthographic_context: HashMap<String, OrthographicContext>,
}

impl TrainingData {
  /// Creates a new, empty data object.
  #[inline(always)]
  pub fn new() -> TrainingData {
    TrainingData {
      ..Default::default()
    }
  }

  /// Check if a token is considered to be an abbreviation.
  #[inline(always)]
  pub fn contains_abbrev(&self, tok: &str) -> bool {
    self.abbrevs.contains(tok)
  }

  /// Insert a newly learned abbreviation.
  #[inline]
  pub(crate) fn insert_abbrev(&mut self, tok: &str) -> bool {
    if !self.contains_abbrev(tok) {
      self.abbrevs.insert(tok.to_lowercase())
    } else {
      false
    }
  }

  /// Removes a learned abbreviation.
  #[inline]
  pub(crate) fn remove_abbrev(&mut self, tok: &str) -> bool {
    self.abbrevs.remove(tok)
  }

  /// Check if a token is considered to be a token that commonly starts a
  /// sentence.
  #[inline(always)]
  pub fn contains_sentence_starter(&self, tok: &str) -> bool {
    self.sentence_starters.contains(tok)
  }

  /// Insert a newly learned word that signifies the start of a sentence.
  #[inline]
  pub(crate) fn insert_sentence_starter(&mut self, tok: &str) -> bool {
    if !self.contains_sentence_starter(tok) {
      self.sentence_starters.insert(tok.to_string())
    } else {
      false
    }
  }

  /// Checks if a pair of words are commonly known to appear together.
  #[inline]
  pub fn contains_collocation(&self, left: &str, right: &str) -> bool {
    self
      .collocations
      .get(left)
      .map(|s| s.contains(right))
      .unwrap_or(false)
  }

  /// Insert a newly learned pair of words that frequently appear together.
  pub(crate) fn insert_collocation(&mut self, left: &str, right: &str) -> bool {
    if !self.collocations.contains_key(left) {
      self.collocations.insert(left.to_string(), HashSet::new());
    }

    if !self.collocations.get(left).unwrap().contains(right) {
      self
        .collocations
        .get_mut(left)
        .unwrap()
        .insert(right.to_string());
      true
    } else {
      false
    }
  }

  /// Insert or update the known orthographic context that a word commonly
  /// appears in.
  #[inline]
  pub(crate) fn insert_orthographic_context(&mut self, tok: &str, ctxt: OrthographicContext) -> bool {
    // `get_mut` isn't allowed here, without adding an unnecessary lifetime
    // qualifier to `tok`.
    match self.orthographic_context.get_mut(tok) {
      Some(c) => {
        *c |= ctxt;
        return false;
      }
      None => (),
    }

    self.orthographic_context.insert(tok.to_string(), ctxt);
    true
  }

  /// Gets the orthographic context for a token. Returns 0 if the token
  /// was not yet encountered.
  #[inline(always)]
  pub fn get_orthographic_context(&self, tok: &str) -> u8 {
    *self.orthographic_context.get(tok).unwrap_or(&0)
  }
}

impl FromStr for TrainingData {
  type Err = &'static str;

  /// Deserializes JSON and loads the data into a new TrainingData object.
  fn from_str(s: &str) -> Result<TrainingData, &'static str> {
    let mut obj = match serde_json::from_str::<Value>(s) {
      Ok(Value::Object(obj)) => obj,
      _ => return Err("no json object found containing training data"),
    };
    let mut data: TrainingData = Default::default();

    // Gets a JSON array by key, then runs an action for each element matching a pattern.
    macro_rules! read_json_array_data(
      ($path:expr, $mtch:pat, $act:expr) => (
        match obj.remove($path) {
          Some(Value::Array(arr)) => {
            for x in arr.into_iter() {
              match x {
                $mtch => { $act; }
                _ => ()
              }
            }
          }
          _ => return Err("failed to parse expected path")
        }
      );
    );

    read_json_array_data!("abbrev_types", Value::String(st), data.insert_abbrev(&st[..]));

    read_json_array_data!(
      "sentence_starters",
      Value::String(st),
      data.insert_sentence_starter(&st[..])
    );

    // Collocations arrive as 2-element arrays; pop in reverse, then bucket them.
    read_json_array_data!("collocations", Value::Array(mut ar), {
      match (ar.pop(), ar.pop()) {
        (Some(Value::String(r)), Some(Value::String(l))) => data
          .collocations
          .entry(l)
          .or_insert(HashSet::new())
          .insert(r),
        _ => return Err("failed to parse collocations section"),
      };
    });

    match obj.remove("ortho_context") {
      Some(Value::Object(obj)) => {
        for (k, ctxt) in obj.into_iter() {
          ctxt
            .as_u64()
            .map(|c| data.orthographic_context.insert(k, c as u8));
        }
      }
      _ => return Err("failed to parse orthographic context section"),
    }

    Ok(data)
  }
}


// Macro for generating functions to load precompiled data.
macro_rules! preloaded_data(
  ($lang:ident, $file:expr) => (
    impl TrainingData {
      #[inline] #[allow(missing_docs)] pub fn $lang() -> TrainingData {
        FromStr::from_str(include_str!($file)).unwrap()
      }
    }
  )
);

#[cfg(feature = "lang-czech")]
preloaded_data!(czech, "data/czech.json");
#[cfg(feature = "lang-danish")]
preloaded_data!(danish, "data/danish.json");
#[cfg(feature = "lang-dutch")]
preloaded_data!(dutch, "data/dutch.json");
#[cfg(feature = "lang-english")]
preloaded_data!(english, "data/english.json");
#[cfg(feature = "lang-estonian")]
preloaded_data!(estonian, "data/estonian.json");
#[cfg(feature = "lang-finnish")]
preloaded_data!(finnish, "data/finnish.json");
#[cfg(feature = "lang-french")]
preloaded_data!(french, "data/french.json");
#[cfg(feature = "lang-german")]
preloaded_data!(german, "data/german.json");
#[cfg(feature = "lang-greek")]
preloaded_data!(greek, "data/greek.json");
#[cfg(feature = "lang-italian")]
preloaded_data!(italian, "data/italian.json");
#[cfg(feature = "lang-norwegian")]
preloaded_data!(norwegian, "data/norwegian.json");
#[cfg(feature = "lang-polish")]
preloaded_data!(polish, "data/polish.json");
#[cfg(feature = "lang-portuguese")]
preloaded_data!(portuguese, "data/portuguese.json");
#[cfg(feature = "lang-slovene")]
preloaded_data!(slovene, "data/slovene.json");
#[cfg(feature = "lang-spanish")]
preloaded_data!(spanish, "data/spanish.json");
#[cfg(feature = "lang-swedish")]
preloaded_data!(swedish, "data/swedish.json");
#[cfg(feature = "lang-turkish")]
preloaded_data!(turkish, "data/turkish.json");
