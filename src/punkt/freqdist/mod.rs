// Copyright 2016 rust-freqdist Developers
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Implementation of a Frequency Distribution in Rust. Keeps track of how many
//! times an object appears in a larger context (for example, how many times a
//! word appears in a piece of text). The underlying data structure of the
//! Frequency Distribution is a HashMap, so the object that is being counted
//! must be hashable.
//!
//! ```

#![warn(missing_docs)]

use std::ops::Index;
use std::default::Default;
use std::iter::{FromIterator, IntoIterator};
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher, RandomState};
use hashbrown::HashMap;
use hashbrown::hash_map::{Keys, IntoIter, Iter};


static ZERO: usize = 0;


#[allow(missing_docs)]
pub struct FrequencyDistribution<K, S = RandomState> {
    hashmap: HashMap<K, usize, S>,
    sum_counts: usize,
}

#[allow(unused)]
impl<K, H, S> FrequencyDistribution<K, S>
where
    K: Eq + Hash,
    H: Hasher,
    S: BuildHasher<Hasher = H>,
{
    /// Creates a new FrequencyDistrbution with a hasher and size, where
    /// the size is known or can be estimated.
    ///
    #[inline]
    pub fn with_capacity_and_hasher(size: usize, state: S) -> FrequencyDistribution<K, S> {
        FrequencyDistribution {
            hashmap: HashMap::with_capacity_and_hasher(size, state),
            sum_counts: 0,
        }
    }

    /// Creates a new FrequencyDistribution with a hasher and default size.
    ///
    #[inline]
    pub fn with_hasher(state: S) -> FrequencyDistribution<K, S> {
        FrequencyDistribution {
            hashmap: HashMap::with_hasher(state),
            sum_counts: 0,
        }
    }

    /// Iterator over the keys.
    ///
    #[inline]
    pub fn keys(&self) -> Keys<'_, K, usize> {
        self.hashmap.keys()
    }

    /// Iterator over the key, frequency pairs.
    ///
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, usize> {
        self.hashmap.iter()
    }

    #[inline]
    pub fn iter_non_zero(&self) -> NonZeroKeysIter<'_, K> {
        NonZeroKeysIter { iter: self.iter() }
    }

    /// Sum of the total number of items counted thus far.
    ///
    #[inline]
    pub fn sum_counts(&self) -> usize {
        self.sum_counts
    }

    /// Returns the number of entries in the distribution
    ///
    #[inline]
    pub fn len(&self) -> usize {
        self.hashmap.len()
    }

    /// Gets the frequency in which the key occurs.
    #[inline]
    pub fn get<Q: ?Sized>(&self, k: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self[k]
    }

    /// Clears the counts of all keys and clears all keys from
    /// the distribution.
    ///
    #[inline]
    pub fn clear(&mut self) {
        self.hashmap.clear()
    }

    /// Updates the frequency of the value found with the key if it
    /// already exists. Otherwise, inserts the key sizeo the hashmap,
    /// and sets its frequency to 1.
    ///
    #[inline]
    pub fn insert(&mut self, k: K) {
        self.insert_or_incr_by(k, 1);
    }

    /// Removes an item and its associated counts.
    ///
    #[inline]
    pub fn remove<Q: ?Sized>(&mut self, k: &Q)
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        match self.hashmap.remove(k) {
            Some(count) => self.sum_counts -= count,
            None => (),
        }
    }

    /// Inserts a value sizeo the hashmap if it does not exist with a new quantity
    /// specified by the increment. If the value already exists, increments by
    /// the specified amount.
    ///
    #[inline]
    fn insert_or_incr_by(&mut self, k: K, incr: usize) {
        *self.hashmap.entry(k).or_insert(0) += incr;

        self.sum_counts += incr;
    }
}

#[allow(unused)]
impl<K, H, S> FrequencyDistribution<K, S>
where
    K: Eq + Hash,
    H: Hasher + Default,
    S: BuildHasher<Hasher = H> + Default,
{
    /// Creates a new FrequencyDistribution where the size of the
    /// HashMap is unknown.
    ///
    #[inline]
    pub fn new() -> FrequencyDistribution<K, S> {
        FrequencyDistribution::with_hasher(Default::default())
    }

    /// Creates a new FrequencyDistribution where the size of the HashMap
    /// is known, or a estimate can be made.
    ///
    #[inline]
    pub fn with_capacity(size: usize) -> FrequencyDistribution<K, S> {
        FrequencyDistribution::with_capacity_and_hasher(size, Default::default())
    }
}

impl<K, H, S> Default for FrequencyDistribution<K, S>
where
    K: Eq + Hash,
    H: Hasher + Default,
    S: BuildHasher<Hasher = H> + Default,
{
    /// Creates a default FrequencyDistribution.
    ///
    #[inline]
    fn default() -> FrequencyDistribution<K, S> {
        FrequencyDistribution::new()
    }
}

impl<K, H, S> FromIterator<(K, usize)> for FrequencyDistribution<K, S>
where
    K: Eq + Hash,
    H: Hasher,
    S: BuildHasher<Hasher = H>
    + Default,
{
    fn from_iter<T>(iter: T) -> FrequencyDistribution<K, S>
    where
        T: IntoIterator<Item = (K, usize)>,
    {
        let iterator = iter.into_iter();
        let mut fdist = if iterator.size_hint().1.is_some() {
            FrequencyDistribution::with_capacity_and_hasher(
                iterator.size_hint().1.unwrap(),
                Default::default(),
            )
        } else {
            FrequencyDistribution::with_capacity_and_hasher(iterator.size_hint().0, Default::default())
        };

        for (k, freq) in iterator {
            fdist.insert_or_incr_by(k, freq);
        }

        fdist
    }
}

impl<K, H, S> Extend<(K, usize)> for FrequencyDistribution<K, S>
where
    K: Eq + Hash,
    H: Hasher,
    S: BuildHasher<Hasher = H>,
{
    /// Extends the hashmap by adding the keys or updating the frequencies of the keys.
    ///
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (K, usize)>,
    {
        for (k, freq) in iter.into_iter() {
            self.insert_or_incr_by(k, freq);
        }
    }
}

impl<K, H, S> IntoIterator for FrequencyDistribution<K, S>
where
    K: Eq + Hash,
    H: Hasher,
    S: BuildHasher<Hasher = H>,
{
    type Item = (K, usize);
    type IntoIter = IntoIter<K, usize>;

    /// Consumes the distribution, and creates an iterator over the
    /// (Key, Quantity: usize) pairs.
    ///
    #[inline]
    fn into_iter(self) -> IntoIter<K, usize> {
        self.hashmap.into_iter()
    }
}

impl<'a, K, H, S, Q: ?Sized> Index<&'a Q> for FrequencyDistribution<K, S>
where
    K: Eq + Hash + Borrow<Q>,
    H: Hasher,
    S: BuildHasher<Hasher = H>,
    Q: Eq + Hash,
{
    type Output = usize;

    #[inline]
    fn index<'b>(&'b self, index: &Q) -> &'b usize {
        self.hashmap.get(index).unwrap_or(&ZERO)
    }
}

/// Iterator over entries with non-zero quantities.
///
pub struct NonZeroKeysIter<'a, K: 'a> {
    iter: Iter<'a, K, usize>,
}

impl<'a, K: 'a> Iterator for NonZeroKeysIter<'a, K> {
    type Item = &'a K;

    #[inline]
    fn next(&mut self) -> Option<&'a K> {
        loop {
            match self.iter.next() {
                Some((k, c)) if *c > 0 => return Some(k),
                None => return None,
                _ => (),
            }
        }
    }
}




