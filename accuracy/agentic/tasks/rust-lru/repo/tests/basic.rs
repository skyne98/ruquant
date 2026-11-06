use lru::LruCache;

#[test]
fn put_get() {
    let mut c = LruCache::new(2);
    c.put("a", 1);
    c.put("b", 2);
    assert_eq!(c.get(&"a"), Some(&1));
    c.put("c", 3); // evicts "b"
    assert_eq!(c.get(&"b"), None);
    assert_eq!(c.len(), 2);
}
