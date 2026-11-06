use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    UnexpectedChar(usize),
    UnexpectedEnd,
    UnexpectedToken(usize),
    UnknownVariable(String),
    UnknownFunction(String),
    Arity(String, usize),
    DivisionByZero,
}

#[derive(Debug, Clone, PartialEq)]
enum Tok { Num(f64), Id(String), Op(char) }

fn lex(src: &str) -> Result<Vec<(Tok, usize)>, Error> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() { i += 1; continue; }
        let start = i;
        if c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            while i < b.len() && b[i].is_ascii_digit() { i += 1; }
            if i < b.len() && b[i] == b'.' { i += 1; while i < b.len() && b[i].is_ascii_digit() { i += 1; } }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let mut j = i + 1;
                if j < b.len() && (b[j] == b'+' || b[j] == b'-') { j += 1; }
                if j < b.len() && b[j].is_ascii_digit() {
                    while j < b.len() && b[j].is_ascii_digit() { j += 1; }
                    i = j;
                }
            }
            out.push((Tok::Num(src[start..i].parse().unwrap()), start));
        } else if c.is_ascii_alphabetic() || c == b'_' {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') { i += 1; }
            out.push((Tok::Id(src[start..i].to_string()), start));
        } else if b"+-*/%^(),".contains(&c) {
            out.push((Tok::Op(c as char), start));
            i += 1;
        } else {
            return Err(Error::UnexpectedChar(start));
        }
    }
    Ok(out)
}

enum Ast { Num(f64), Var(String), Neg(Box<Ast>), Bin(char, Box<Ast>, Box<Ast>), Call(String, Vec<Ast>) }

struct P { t: Vec<(Tok, usize)>, i: usize }

impl P {
    fn peek(&self) -> Option<&Tok> { self.t.get(self.i).map(|x| &x.0) }
    fn is_op(&self, c: char) -> bool { self.peek() == Some(&Tok::Op(c)) }
    fn err_here(&self) -> Error { match self.t.get(self.i) { Some((_, p)) => Error::UnexpectedToken(*p), None => Error::UnexpectedEnd } }
    fn expect(&mut self, c: char) -> Result<(), Error> { if self.is_op(c) { self.i += 1; Ok(()) } else { Err(self.err_here()) } }
    fn sum(&mut self) -> Result<Ast, Error> {
        let mut l = self.term()?;
        while let Some(Tok::Op(c @ ('+' | '-'))) = self.peek().cloned() { self.i += 1; l = Ast::Bin(c, Box::new(l), Box::new(self.term()?)); }
        Ok(l)
    }
    fn term(&mut self) -> Result<Ast, Error> {
        let mut l = self.unary()?;
        while let Some(Tok::Op(c @ ('*' | '/' | '%'))) = self.peek().cloned() { self.i += 1; l = Ast::Bin(c, Box::new(l), Box::new(self.unary()?)); }
        Ok(l)
    }
    fn unary(&mut self) -> Result<Ast, Error> {
        if self.is_op('-') { self.i += 1; return Ok(Ast::Neg(Box::new(self.unary()?))); }
        if self.is_op('+') { self.i += 1; return self.unary(); }
        self.power()
    }
    fn power(&mut self) -> Result<Ast, Error> {
        let base = self.atom()?;
        if self.is_op('^') { self.i += 1; return Ok(Ast::Bin('^', Box::new(base), Box::new(self.unary()?))); }
        Ok(base)
    }
    fn atom(&mut self) -> Result<Ast, Error> {
        match self.peek().cloned() {
            Some(Tok::Num(v)) => { self.i += 1; Ok(Ast::Num(v)) }
            Some(Tok::Id(name)) => {
                self.i += 1;
                if self.is_op('(') {
                    self.i += 1;
                    let mut args = Vec::new();
                    if !self.is_op(')') {
                        loop { args.push(self.sum()?); if self.is_op(',') { self.i += 1; } else { break; } }
                    }
                    self.expect(')')?;
                    Ok(Ast::Call(name, args))
                } else { Ok(Ast::Var(name)) }
            }
            Some(Tok::Op('(')) => { self.i += 1; let e = self.sum()?; self.expect(')')?; Ok(e) }
            _ => Err(self.err_here()),
        }
    }
}

fn ev(a: &Ast, vars: &HashMap<String, f64>) -> Result<f64, Error> {
    Ok(match a {
        Ast::Num(v) => *v,
        Ast::Var(n) => *vars.get(n).ok_or_else(|| Error::UnknownVariable(n.clone()))?,
        Ast::Neg(x) => -ev(x, vars)?,
        Ast::Bin(op, l, r) => {
            let (l, r) = (ev(l, vars)?, ev(r, vars)?);
            match op {
                '+' => l + r, '-' => l - r, '*' => l * r,
                '/' => { if r == 0.0 { return Err(Error::DivisionByZero) } l / r }
                '%' => { if r == 0.0 { return Err(Error::DivisionByZero) } l % r }
                _ => l.powf(r),
            }
        }
        Ast::Call(n, args) => {
            let vals = args.iter().map(|x| ev(x, vars)).collect::<Result<Vec<_>, _>>()?;
            match n.as_str() {
                "abs" | "sqrt" => { if vals.len() != 1 { return Err(Error::Arity(n.clone(), vals.len())) } if n == "abs" { vals[0].abs() } else { vals[0].sqrt() } }
                "min" | "max" => {
                    if vals.is_empty() { return Err(Error::Arity(n.clone(), 0)) }
                    vals.into_iter().reduce(|a, b| if n == "min" { a.min(b) } else { a.max(b) }).unwrap()
                }
                _ => return Err(Error::UnknownFunction(n.clone())),
            }
        }
    })
}

pub fn eval(src: &str, vars: &HashMap<String, f64>) -> Result<f64, Error> {
    let mut p = P { t: lex(src)?, i: 0 };
    let ast = p.sum()?;
    if p.i < p.t.len() { return Err(Error::UnexpectedToken(p.t[p.i].1)); }
    ev(&ast, vars)
}
