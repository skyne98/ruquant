//! A least-recently-used cache with a fixed capacity (reference: slab-backed doubly linked list + HashMap).

use std::collections::HashMap;
use std::hash::Hash;

const NIL: usize = usize::MAX;

struct Node<K, V> {
    key: K,
    value: V,
    prev: usize,
    next: usize,
}

pub struct LruCache<K, V> {
    map: HashMap<K, usize>,
    nodes: Vec<Option<Node<K, V>>>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
    cap: usize,
}

impl<K: Hash + Eq + Clone, V> LruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        LruCache { map: HashMap::new(), nodes: Vec::new(), free: Vec::new(), head: NIL, tail: NIL, cap: capacity }
    }
    fn node(&self, i: usize) -> &Node<K, V> {
        self.nodes[i].as_ref().unwrap()
    }
    fn node_mut(&mut self, i: usize) -> &mut Node<K, V> {
        self.nodes[i].as_mut().unwrap()
    }
    fn unlink(&mut self, i: usize) {
        let (p, n) = (self.node(i).prev, self.node(i).next);
        if p != NIL { self.node_mut(p).next = n } else { self.head = n }
        if n != NIL { self.node_mut(n).prev = p } else { self.tail = p }
    }
    fn push_front(&mut self, i: usize) {
        let h = self.head;
        { let n = self.node_mut(i); n.prev = NIL; n.next = h; }
        if h != NIL { self.node_mut(h).prev = i } else { self.tail = i }
        self.head = i;
    }
    fn touch(&mut self, i: usize) {
        if self.head != i { self.unlink(i); self.push_front(i); }
    }
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let i = *self.map.get(key)?;
        self.touch(i);
        Some(&self.node(i).value)
    }
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let i = *self.map.get(key)?;
        self.touch(i);
        Some(&mut self.node_mut(i).value)
    }
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|&i| &self.node(i).value)
    }
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        if let Some(&i) = self.map.get(&key) {
            self.touch(i);
            return Some(std::mem::replace(&mut self.node_mut(i).value, value));
        }
        if self.map.len() == self.cap { self.pop_lru(); }
        let node = Node { key: key.clone(), value, prev: NIL, next: NIL };
        let i = match self.free.pop() { Some(i) => { self.nodes[i] = Some(node); i } None => { self.nodes.push(Some(node)); self.nodes.len() - 1 } };
        self.map.insert(key, i);
        self.push_front(i);
        None
    }
    fn take(&mut self, i: usize) -> (K, V) {
        self.unlink(i);
        let n = self.nodes[i].take().unwrap();
        self.free.push(i);
        self.map.remove(&n.key);
        (n.key, n.value)
    }
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let i = *self.map.get(key)?;
        Some(self.take(i).1)
    }
    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        if self.tail == NIL { None } else { Some(self.take(self.tail)) }
    }
    pub fn contains(&self, key: &K) -> bool { self.map.contains_key(key) }
    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }
    pub fn capacity(&self) -> usize { self.cap }
    pub fn resize(&mut self, capacity: usize) {
        assert!(capacity > 0);
        self.cap = capacity;
        while self.map.len() > capacity { self.pop_lru(); }
    }
    pub fn clear(&mut self) {
        self.map.clear(); self.nodes.clear(); self.free.clear(); self.head = NIL; self.tail = NIL;
    }
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        let mut i = self.head;
        std::iter::from_fn(move || {
            if i == NIL { return None }
            let n = self.node(i);
            i = n.next;
            Some((&n.key, &n.value))
        })
    }
}
