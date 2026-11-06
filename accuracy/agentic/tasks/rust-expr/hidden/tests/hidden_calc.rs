use calc::{eval, Error};
use std::collections::HashMap;

fn e(s: &str) -> Result<f64, Error> {
    let mut v = HashMap::new();
    v.insert("x".to_string(), 3.0);
    v.insert("_y2".to_string(), -0.5);
    eval(s, &v)
}

fn close(s: &str, want: f64) {
    let got = e(s).unwrap_or_else(|err| panic!("{s}: {err:?}"));
    assert!((got - want).abs() < 1e-9, "{s}: got {got}, want {want}");
}

#[test]
fn numbers() {
    close("3", 3.0);
    close("3.25", 3.25);
    close(".5", 0.5);
    close("2e3", 2000.0);
    close("1.5E-2", 0.015);
    close("  7  ", 7.0);
}

#[test]
fn precedence_and_associativity() {
    close("1 + 2 * 3", 7.0);
    close("(1 + 2) * 3", 9.0);
    close("10 - 4 - 3", 3.0);
    close("2 * 9 / 3", 6.0);
    close("100 / 10 / 5", 2.0);
    close("7 % 4 * 2", 6.0);
    close("-7 % 4", -3.0);
    close("2 ^ 3 ^ 2", 512.0);
    close("-2 ^ 2", -4.0);
    close("2 ^ -1", 0.5);
    close("--2", 2.0);
    close("-+-2", 2.0);
    close("2 * -3", -6.0);
    close("(2 ^ 2) ^ 3", 64.0);
}

#[test]
fn variables_and_functions() {
    close("x * 2", 6.0);
    close("_y2 + x", 2.5);
    close("abs(_y2)", 0.5);
    close("sqrt(16) + 1", 5.0);
    close("min(4, x, 9)", 3.0);
    close("max(x)", 3.0);
    close("max(1, min(2, 3) * 4, -5)", 8.0);
    close("-abs(-2)^2", -4.0);
}

#[test]
fn lexing_errors() {
    assert_eq!(e("2 $ 3"), Err(Error::UnexpectedChar(2)));
    assert_eq!(e("1 + #"), Err(Error::UnexpectedChar(4)));
}

#[test]
fn parse_errors() {
    assert_eq!(e(""), Err(Error::UnexpectedEnd));
    assert_eq!(e("2 +"), Err(Error::UnexpectedEnd));
    assert_eq!(e("(1"), Err(Error::UnexpectedEnd));
    assert_eq!(e("min(1,"), Err(Error::UnexpectedEnd));
    assert_eq!(e("2 3"), Err(Error::UnexpectedToken(2)));
    assert_eq!(e(")"), Err(Error::UnexpectedToken(0)));
    assert_eq!(e("(1))"), Err(Error::UnexpectedToken(3)));
    assert_eq!(e("1 + * 2"), Err(Error::UnexpectedToken(4)));
    assert_eq!(e("abs 2"), Err(Error::UnexpectedToken(4)));
}

#[test]
fn evaluation_errors() {
    assert_eq!(e("1 / 0"), Err(Error::DivisionByZero));
    assert_eq!(e("5 % (x - 3)"), Err(Error::DivisionByZero));
    assert_eq!(e("y + 1"), Err(Error::UnknownVariable("y".into())));
    assert_eq!(e("foo(1)"), Err(Error::UnknownFunction("foo".into())));
    assert_eq!(e("abs(1, 2)"), Err(Error::Arity("abs".into(), 2)));
    assert_eq!(e("sqrt()"), Err(Error::Arity("sqrt".into(), 0)));
    assert_eq!(e("min()"), Err(Error::Arity("min".into(), 0)));
}

#[test]
fn parse_errors_come_before_evaluation_errors() {
    assert_eq!(e("1 / 0 )"), Err(Error::UnexpectedToken(6)));
    assert_eq!(e("y +"), Err(Error::UnexpectedEnd));
}
