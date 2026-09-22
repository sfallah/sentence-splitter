//! Training new Punkt models from a corpus.
//!
//! Vendored from upstream `rust-punkt` and **unsound**: it casts `&TrainingData` and
//! `&Token` to `&mut` to mutate them while an iterator still borrows them, which is
//! undefined behaviour. rustc's `invalid_reference_casting` lint rejects it; cargo only
//! hides that upstream because it caps lints in dependencies. Fixing it properly means
//! preserving the mutate-while-iterating semantics the algorithm relies on, so it is
//! left as-is behind this off-by-default feature rather than silently changing how
//! training behaves. Loading a pretrained model never touches this code.
#![allow(unsafe_code, invalid_reference_casting)]

// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::min;
use std::marker::PhantomData;
use crate::punkt::freqdist::FrequencyDistribution;

use crate::punkt::prelude::{
  DefinesNonPrefixCharacters, DefinesNonWordCharacters, OrthographicContext, OrthographyPosition,
  TrainerParameters,
};
use crate::punkt::token::Token;
use crate::punkt::trainer::{Collocation, TrainingData};
use crate::punkt::tokenizer::WordTokenizer;
use crate::punkt::util;

/// A collocation is any pair of words that has a high likelihood of appearing
/// together.

/// A trainer will build data about abbreviations, sentence starters,
/// collocations, and context that tokens appear in. The data is
/// used by the sentence tokenizer to determine if a period is likely
/// part of an abbreviation, or actually marks the termination of a sentence.
pub struct Trainer<P> {
  params: PhantomData<P>,
}

impl<P> Trainer<P>
where
  P: TrainerParameters + DefinesNonPrefixCharacters + DefinesNonWordCharacters,
{
  /// Creates a new Trainer.
  #[inline(always)]
  pub fn new() -> Trainer<P> {
    Trainer {
      params: PhantomData,
    }
  }

  /// Train on a document. Does tokenization using a WordTokenizer.
  pub fn train(&self, doc: &str, data: &mut TrainingData) {
    let mut period_token_count: usize = 0;
    let mut sentence_break_count: usize = 0;
    let tokens: Vec<Token> = WordTokenizer::<P>::new(doc).collect();
    let mut type_fdist: FrequencyDistribution<&str> = FrequencyDistribution::new();
    let mut collocation_fdist = FrequencyDistribution::new();
    let mut sentence_starter_fdist = FrequencyDistribution::new();

    for t in tokens.iter() {
      if t.has_final_period() {
        period_token_count += 1
      }
      type_fdist.insert(t.typ());
    }

    // Iterate through to see if any tokens need to be reclassified as an
    // abbreviation or removed as an abbreviation.
    {
      let reclassify_iter: ReclassifyIterator<_, P> = ReclassifyIterator {
        iter: tokens.iter(),
        data: data,
        period_token_count: period_token_count,
        type_fdist: &mut type_fdist,
        params: PhantomData,
      };

      for (t, score) in reclassify_iter {
        if score >= P::ABBREV_LOWER_BOUND {
          if t.has_final_period() {
            unsafe {
              (&mut *(data as *const TrainingData as *mut TrainingData))
                .insert_abbrev(t.typ_without_period());
            }
          }
        } else {
          if !t.has_final_period() {
            unsafe {
              (&mut *(data as *const TrainingData as *mut TrainingData))
                .remove_abbrev(t.typ_without_period());
            }
          }
        }
      }
    }

    // Annotating the tokens requires an unsafe block, but it won't modify any pointers,
    // just will modify some flags on the tokens.
    for t in tokens.iter() {
      unsafe {
        util::annotate_first_pass::<P>(&mut *(t as *const Token as *mut Token), data);
      }
    }

    // Update or insert the orthographic context of all tokens in the document.
    {
      let token_with_context_iter = TokenWithContextIterator {
        iter: tokens.iter(),
        ctxt: OrthographyPosition::Internal,
      };

      for (t, ctxt) in token_with_context_iter {
        if ctxt != 0 {
          data.insert_orthographic_context(t.typ_without_break_or_period(), ctxt);
        }
      }
    }

    // Order matters! Sentence break checks are dependent on whether or not
    // the token is an abbreviation. Must come after the first pass annotation!
    for t in tokens.iter() {
      if t.is_sentence_break() {
        sentence_break_count += 1;
      }
    }

    // Iterate over tokens, and determine if they're abbreviations or if they
    // are potential sentence starters or potential collocations.
    {
      let consecutive_token_iter = ConsecutiveItemIterator {
        iter: tokens.iter(),
        last: None,
      };

      for (lt, rt) in consecutive_token_iter {
        match rt {
          Some(cur) if lt.has_final_period() => {
            if is_rare_abbrev_type::<P>(&data, &type_fdist, lt, cur) {
              data.insert_abbrev(lt.typ_without_period());
            }

            if is_potential_sentence_starter(cur, lt) {
              sentence_starter_fdist.insert(cur);
            }

            if is_potential_collocation::<P>(lt, cur) {
              collocation_fdist.insert(Collocation::new(lt, cur));
            }
          }
          _ => (),
        }
      }
    }

    {
      let ss_iter: PotentialSentenceStartersIterator<_, P> = PotentialSentenceStartersIterator {
        iter: sentence_starter_fdist.keys(),
        sentence_break_count: sentence_break_count,
        type_fdist: &type_fdist,
        sentence_starter_fdist: &sentence_starter_fdist,
        params: PhantomData,
      };

      for (tok, _) in ss_iter {
        data.insert_sentence_starter(tok.typ());
      }
    }

    {
      let clc_iter: PotentialCollocationsIterator<_, P> = PotentialCollocationsIterator {
        iter: collocation_fdist.keys(),
        data: &data,
        type_fdist: &type_fdist,
        collocation_fdist: &collocation_fdist,
        params: PhantomData,
      };

      for (col, _) in clc_iter {
        unsafe {
          (&mut *(data as *const TrainingData as *mut TrainingData)).insert_collocation(
            col.left().typ_without_period(),
            col.right().typ_without_break_or_period(),
          );
        }
      }
    }
  }
}

fn is_rare_abbrev_type<P>(
  data: &TrainingData,
  type_fdist: &FrequencyDistribution<&str>,
  tok0: &Token,
  tok1: &Token,
) -> bool
where
  P: TrainerParameters,
{
  use crate::punkt::prelude::{BEG_UC, MID_UC};

  if tok0.is_abbrev() || !tok0.is_sentence_break() {
    false
  } else {
    let key = tok0.typ_without_break_or_period();
    let count = (type_fdist[key] + type_fdist[&key[..key.len() - 1]]) as f64;

    // Already an abbreviation...
    if data.contains_abbrev(tok0.typ()) || count >= P::ABBREV_UPPER_BOUND {
      false
    } else if P::is_internal_punctuation(&tok1.typ().chars().next().unwrap()) {
      true
    } else if tok1.is_lowercase() {
      let ctxt = data.get_orthographic_context(tok1.typ_without_break_or_period());

      if (ctxt & BEG_UC > 0) && !(ctxt & MID_UC > 0) {
        true
      } else {
        false
      }
    } else {
      false
    }
  }
}

#[inline(always)]
fn is_potential_sentence_starter(cur: &Token, prev: &Token) -> bool {
  prev.is_sentence_break() && !(prev.is_numeric() || prev.is_initial()) && cur.is_alphabetic()
}

#[inline(always)]
fn is_potential_collocation<P>(tok0: &Token, tok1: &Token) -> bool
where
  P: TrainerParameters,
{
  P::INCLUDE_ALL_COLLOCATIONS
    || (P::INCLUDE_ABBREV_COLLOCATIONS && tok0.is_abbrev())
    || (tok0.is_sentence_break() && (tok0.is_numeric() || tok0.is_initial()))
      && tok0.is_non_punct()
      && tok1.is_non_punct()
}

/// Iterates over every token from the supplied iterator. Only returns
/// the ones that are 'not obviously' abbreviations. Also returns the associated
/// score of that token.
struct ReclassifyIterator<'b, I, P> {
  iter: I,
  data: &'b TrainingData,
  period_token_count: usize,
  type_fdist: &'b FrequencyDistribution<&'b str>,
  params: PhantomData<P>,
}

impl<'b, I, P> Iterator for ReclassifyIterator<'b, I, P>
where
  I: Iterator<Item = &'b Token>,
  P: TrainerParameters,
{
  type Item = (&'b Token, f64);

  #[inline]
  fn next(&mut self) -> Option<Self::Item> {
    while let Some(t) = self.iter.next() {
      if !t.is_non_punct() || t.is_numeric() {
        continue;
      }

      if t.has_final_period() {
        if self.data.contains_abbrev(t.typ()) {
          continue;
        }
      } else {
        if !self.data.contains_abbrev(t.typ()) {
          continue;
        }
      }

      let num_periods =
        t.typ_without_period()
          .chars()
          .fold(0, |acc, c| if c == '.' { acc + 1 } else { acc })
          + 1;
      let num_nonperiods = t.typ_without_period().chars().count() - num_periods + 1;

      let count_with_period = self.type_fdist.get(t.typ_with_period());
      let count_without_period = self.type_fdist.get(t.typ_without_period());

      let likelihood = util::dunning_log_likelihood(
        (count_with_period + count_without_period) as f64,
        self.period_token_count as f64,
        count_with_period as f64,
        self.type_fdist.sum_counts() as f64,
      );

      let f_length = (-(num_nonperiods as f64)).exp();
      let f_penalty = if P::IGNORE_ABBREV_PENALTY {
        0f64
      } else {
        (num_nonperiods as f64).powi(-(count_without_period as i32))
      };

      let score = likelihood * f_length * f_penalty * (num_periods as f64);

      return Some((t, score));
    }

    None
  }
}

struct TokenWithContextIterator<I> {
  iter: I,
  ctxt: OrthographyPosition,
}

impl<'a, I> Iterator for TokenWithContextIterator<I>
where
  I: Iterator<Item = &'a Token>,
{
  type Item = (&'a Token, OrthographicContext);

  #[inline]
  fn next(&mut self) -> Option<(&'a Token, OrthographicContext)> {
    match self.iter.next() {
      Some(t) => {
        if t.is_paragraph_start() && self.ctxt != OrthographyPosition::Unknown {
          self.ctxt = OrthographyPosition::Initial;
        }

        if t.is_newline_start() && self.ctxt == OrthographyPosition::Internal {
          self.ctxt = OrthographyPosition::Unknown;
        }

        let flag = *crate::punkt::prelude::ORTHO_MAP
          .get(&(self.ctxt.as_byte() | t.first_case().as_byte()))
          .unwrap_or(&0);

        if t.is_sentence_break() {
          if !(t.is_numeric() || t.is_initial()) {
            self.ctxt = OrthographyPosition::Initial;
          } else {
            self.ctxt = OrthographyPosition::Unknown;
          }
        } else if t.is_ellipsis() || t.is_abbrev() {
          self.ctxt = OrthographyPosition::Unknown;
        } else {
          self.ctxt = OrthographyPosition::Internal;
        }

        Some((t, flag))
      }
      None => None,
    }
  }
}

struct PotentialCollocationsIterator<'b, I, P> {
  iter: I,
  data: &'b TrainingData,
  type_fdist: &'b FrequencyDistribution<&'b str>,
  collocation_fdist: &'b FrequencyDistribution<Collocation<&'b Token>>,
  params: PhantomData<P>,
}

impl<'a, 'b, I, P> Iterator for PotentialCollocationsIterator<'b, I, P>
where
  I: Iterator<Item = &'a Collocation<&'a Token>>,
  P: TrainerParameters,
{
  type Item = (&'a Collocation<&'a Token>, f64);

  #[inline]
  fn next(&mut self) -> Option<(&'a Collocation<&'a Token>, f64)> {
    while let Some(col) = self.iter.next() {
      if self
        .data
        .contains_sentence_starter(col.right().typ_without_break_or_period())
      {
        continue;
      }

      let count = self.collocation_fdist.get(col);

      let left_count = self.type_fdist.get(col.left().typ_without_period())
        + self.type_fdist.get(col.left().typ_with_period());
      let right_count = self.type_fdist.get(col.right().typ_without_period())
        + self.type_fdist.get(col.right().typ_with_period());

      if left_count > 1
        && right_count > 1
        && P::COLLOCATION_FREQUENCY_LOWER_BOUND < count as f64
        && count <= min(left_count, right_count)
      {
        let likelihood = util::col_log_likelihood(
          left_count as f64,
          right_count as f64,
          count as f64,
          self.type_fdist.sum_counts() as f64,
        );

        if likelihood >= P::COLLOCATION_LOWER_BOUND
          && (self.type_fdist.sum_counts() as f64 / left_count as f64)
            > (right_count as f64 / count as f64)
        {
          return Some((col, likelihood));
        }
      }
    }

    None
  }
}

struct PotentialSentenceStartersIterator<'b, I, P> {
  iter: I,
  sentence_break_count: usize,
  type_fdist: &'b FrequencyDistribution<&'b str>,
  sentence_starter_fdist: &'b FrequencyDistribution<&'b Token>,
  params: PhantomData<P>,
}

impl<'a, 'b, I, P> Iterator for PotentialSentenceStartersIterator<'b, I, P>
where
  I: Iterator<Item = &'a &'a Token>,
  P: TrainerParameters,
{
  type Item = (&'a Token, f64);

  #[inline]
  fn next(&mut self) -> Option<(&'a Token, f64)> {
    while let Some(tok) = self.iter.next() {
      let ss_count = self.sentence_starter_fdist.get(tok);
      let typ_count =
        self.type_fdist.get(tok.typ_with_period()) + self.type_fdist.get(tok.typ_without_period());

      if typ_count < ss_count {
        continue;
      }

      let likelihood = util::col_log_likelihood(
        self.sentence_break_count as f64,
        typ_count as f64,
        ss_count as f64,
        self.type_fdist.sum_counts() as f64,
      );

      let ratio = self.type_fdist.sum_counts() as f64 / self.sentence_break_count as f64;

      if likelihood >= P::SENTENCE_STARTER_LOWER_BOUND
        && ratio > (typ_count as f64 / ss_count as f64)
      {
        return Some((*tok, likelihood));
      }
    }

    None
  }
}

struct ConsecutiveItemIterator<'a, T: 'a, I>
where
  I: Iterator<Item = &'a T>,
{
  iter: I,
  last: Option<&'a T>,
}

impl<'a, T: 'a, I> Iterator for ConsecutiveItemIterator<'a, T, I>
where
  I: Iterator<Item = &'a T>,
{
  type Item = (&'a T, Option<&'a T>);

  #[inline]
  fn next(&mut self) -> Option<(&'a T, Option<&'a T>)> {
    match self.last {
      Some(i) => {
        self.last = self.iter.next();
        Some((i, self.last))
      }
      None => match self.iter.next() {
        Some(i) => {
          self.last = self.iter.next();
          Some((i, self.last))
        }
        None => None,
      },
    }
  }
}
