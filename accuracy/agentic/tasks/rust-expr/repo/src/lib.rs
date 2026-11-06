//! Evaluates arithmetic expressions. See SPEC.md.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A character that can't start a token, at this byte offset.
    UnexpectedChar(usize),
    /// The input ended where an operand or a closing parenthesis was needed.
    UnexpectedEnd,
    /// A token that doesn't fit the grammar, at the byte offset where it starts.
    UnexpectedToken(usize),
    UnknownVariable(String),
    UnknownFunction(String),
    /// A function called with the wrong number of arguments: (name, given).
    Arity(String, usize),
    DivisionByZero,
}

pub fn eval(src: &str, vars: &HashMap<String, f64>) -> Result<f64, Error> {
    todo!()
}
