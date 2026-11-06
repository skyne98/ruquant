use calc::eval;
use std::collections::HashMap;

#[test]
fn smoke() {
    assert_eq!(eval("1 + 2 * 3", &HashMap::new()), Ok(7.0));
}
