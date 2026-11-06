//! A least-recently-used cache with a fixed capacity.

use std::hash::Hash;

/// A map that holds at most `capacity` entries. When a new key is inserted into a full cache, the least recently
/// used entry is evicted. `get` and `put` count as uses; `peek` does not.
///
/// All operations except `iter` and `resize` are O(1) on average.
pub struct LruCache<K, V> {
    _marker: std::marker::PhantomData<(K, V)>,
}

impl<K: Hash + Eq + Clone, V> LruCache<K, V> {
    /// Panics if `capacity` is 0.
    pub fn new(capacity: usize) -> Self {
        todo!()
    }

    /// Returns the value and marks the entry as most recently used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        todo!()
    }

    /// Like `get`, but returns a mutable reference.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        todo!()
    }

    /// Returns the value without changing the order.
    pub fn peek(&self, key: &K) -> Option<&V> {
        todo!()
    }

    /// Inserts or replaces a value and marks it as most recently used. Returns the old value if the key was
    /// present. If the key is new and the cache is full, the least recently used entry is evicted first.
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        todo!()
    }

    /// Removes an entry and returns its value.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        todo!()
    }

    /// Removes and returns the least recently used entry.
    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        todo!()
    }

    pub fn contains(&self, key: &K) -> bool {
        todo!()
    }

    pub fn len(&self) -> usize {
        todo!()
    }

    pub fn is_empty(&self) -> bool {
        todo!()
    }

    pub fn capacity(&self) -> usize {
        todo!()
    }

    /// Changes the capacity, evicting least recently used entries until `len() <= capacity`. Panics if
    /// `capacity` is 0.
    pub fn resize(&mut self, capacity: usize) {
        todo!()
    }

    /// Removes every entry.
    pub fn clear(&mut self) {
        todo!()
    }

    /// Iterates from the most recently used entry to the least recently used one.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        std::iter::empty()
    }
}
