use lru::LruCache;
use std::rc::Rc;
use std::time::Instant;

fn keys(c: &LruCache<i32, i32>) -> Vec<i32> {
    c.iter().map(|(k, _)| *k).collect()
}

#[test]
fn order_and_iter() {
    let mut c = LruCache::new(3);
    c.put(1, 10);
    c.put(2, 20);
    c.put(3, 30);
    assert_eq!(keys(&c), vec![3, 2, 1]);
    c.get(&1);
    assert_eq!(keys(&c), vec![1, 3, 2]);
    c.put(4, 40); // evicts 2
    assert_eq!(keys(&c), vec![4, 1, 3]);
    assert!(!c.contains(&2));
}

#[test]
fn peek_does_not_touch() {
    let mut c = LruCache::new(2);
    c.put(1, 1);
    c.put(2, 2);
    assert_eq!(c.peek(&1), Some(&1));
    c.put(3, 3); // 1 is still the LRU
    assert!(!c.contains(&1));
    assert!(c.contains(&2) && c.contains(&3));
}

#[test]
fn put_existing_returns_old_and_touches() {
    let mut c = LruCache::new(2);
    assert_eq!(c.put(1, 1), None);
    c.put(2, 2);
    assert_eq!(c.put(1, 100), Some(1));
    assert_eq!(c.len(), 2);
    c.put(3, 3); // evicts 2, not 1
    assert_eq!(c.peek(&1), Some(&100));
    assert!(!c.contains(&2));
}

#[test]
fn get_mut_touches() {
    let mut c = LruCache::new(2);
    c.put(1, 1);
    c.put(2, 2);
    *c.get_mut(&1).unwrap() += 5;
    c.put(3, 3);
    assert_eq!(c.peek(&1), Some(&6));
    assert!(!c.contains(&2));
    assert_eq!(c.get_mut(&9), None);
}

#[test]
fn remove_pop_clear() {
    let mut c = LruCache::new(3);
    c.put(1, 1);
    c.put(2, 2);
    c.put(3, 3);
    assert_eq!(c.remove(&2), Some(2));
    assert_eq!(c.remove(&2), None);
    assert_eq!(c.pop_lru(), Some((1, 1)));
    assert_eq!(c.len(), 1);
    c.put(4, 4);
    c.put(5, 5);
    assert_eq!(keys(&c), vec![5, 4, 3]);
    c.clear();
    assert!(c.is_empty());
    assert_eq!(c.pop_lru(), None);
    c.put(6, 6);
    assert_eq!(keys(&c), vec![6]);
}

#[test]
fn resize_evicts_lru() {
    let mut c = LruCache::new(5);
    for i in 0..5 {
        c.put(i, i);
    }
    c.get(&0);
    c.resize(2);
    assert_eq!(c.capacity(), 2);
    assert_eq!(keys(&c), vec![0, 4]);
    c.resize(4);
    c.put(7, 7);
    c.put(8, 8);
    assert_eq!(keys(&c), vec![8, 7, 0, 4]);
}

#[test]
#[should_panic]
fn zero_capacity_panics() {
    let _ = LruCache::<i32, i32>::new(0);
}

#[test]
fn capacity_one() {
    let mut c = LruCache::new(1);
    c.put("a", 1);
    c.put("b", 2);
    assert_eq!(c.get(&"a"), None);
    assert_eq!(c.get(&"b"), Some(&2));
}

#[test]
fn values_are_dropped() {
    let rc = Rc::new(());
    {
        let mut c = LruCache::new(2);
        for i in 0..10 {
            c.put(i, rc.clone());
        }
        c.remove(&9);
        assert_eq!(Rc::strong_count(&rc), 2);
    }
    assert_eq!(Rc::strong_count(&rc), 1);
}

#[test]
fn string_keys() {
    let mut c: LruCache<String, Vec<u8>> = LruCache::new(2);
    c.put("x".to_string(), vec![1]);
    c.put("y".to_string(), vec![2]);
    c.get(&"x".to_string());
    c.put("z".to_string(), vec![3]);
    assert!(c.contains(&"x".to_string()) && !c.contains(&"y".to_string()));
}

#[test]
fn constant_time_operations() {
    // 200k operations on a 50k-entry cache; a linear scan per operation would take minutes.
    let mut c = LruCache::new(50_000);
    let t = Instant::now();
    for i in 0..100_000u64 {
        c.put(i, i);
        let k = i.wrapping_mul(2654435761) % (i + 1);
        c.get(&k);
    }
    assert_eq!(c.len(), 50_000);
    assert!(t.elapsed().as_secs() < 20, "too slow: {:?}", t.elapsed());
}
