# Expression language

`calc::eval(src, vars)` evaluates one expression to an `f64`.

Tokens (whitespace between tokens is ignored):

- numbers: digits with an optional fraction and an optional exponent: `3`, `3.25`, `.5`, `2e3`, `1.5E-2`.
  A number can't start with `e` and can't have a sign (a leading `-` is the unary operator).
- identifiers: a letter or `_`, then letters, digits or `_`. An identifier followed by `(` is a function call,
  otherwise a variable looked up in `vars`.
- operators `+ - * / % ^`, parentheses, and `,` between function arguments.

Precedence, lowest first:

1. `+ -` binary, left associative
2. `* / %` left associative; `%` is the remainder with the sign of the dividend (Rust's `%` on f64)
3. unary `-` and `+` (can repeat: `--2` is 2)
4. `^` right associative, binds tighter than unary minus: `-2^2` is -4, `2^-1` is 0.5 (the exponent of `^` is a
   unary expression), `2^3^2` is 512

Functions: `abs(x)`, `sqrt(x)`, `min(a, ...)`, `max(a, ...)` (min and max take one or more arguments).

Errors (`Error`):

- `UnexpectedChar(pos)`: a character that can't start any token, e.g. `2 $ 3` gives `UnexpectedChar(2)`.
- `UnexpectedEnd`: the input ends early, e.g. `2 +`, `(1`, `min(1,`, or an empty input.
- `UnexpectedToken(pos)`: a token in the wrong place, e.g. `2 3` gives `UnexpectedToken(2)`, `)` gives
  `UnexpectedToken(0)`, `(1))` gives `UnexpectedToken(3)`.
- `UnknownVariable(name)`, `UnknownFunction(name)`.
- `Arity(name, n)`: `abs` and `sqrt` need exactly 1 argument, `min` and `max` at least 1; `n` is the number given.
- `DivisionByZero`: `/` or `%` with a right operand equal to 0.

Positions are byte offsets into `src`. When several errors are present, report the first one found reading left to
right (evaluation errors such as `DivisionByZero` or `UnknownVariable` only come after the whole input has parsed).
