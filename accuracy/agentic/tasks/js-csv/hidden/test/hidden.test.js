import { test } from "node:test";
import assert from "node:assert/strict";
import { parse, stringify } from "../src/index.js";
import { CsvError } from "../src/parse.js";

test("hidden: last record without line break keeps an empty last field", () => {
  assert.deepEqual(parse("a,b,"), [["a", "b", ""]]);
  assert.deepEqual(parse("x\na,"), [["x"], ["a", ""]]);
});

test("hidden: empty quoted field at the end", () => {
  assert.deepEqual(parse('""'), [[""]]);
  assert.deepEqual(parse('a,""'), [["a", ""]]);
});

test("hidden: escaped quotes in the middle, CRLF", () => {
  assert.deepEqual(parse('a,"b ""c"" d"\r\n'), [["a", 'b "c" d']]);
});

test("hidden: CRLF inside quotes is kept", () => {
  assert.deepEqual(parse('"a\r\nb",c\r\n'), [["a\r\nb", "c"]]);
});

test("hidden: no trailing line break", () => {
  assert.deepEqual(parse("a,b\r\n1,2"), [["a", "b"], ["1", "2"]]);
});

test("hidden: other delimiter", () => {
  const rows = [["k", "v;w"], ["1", 'q"']];
  const text = stringify(rows, { delimiter: ";" });
  assert.deepEqual(parse(text, { delimiter: ";" }), rows);
  assert.deepEqual(parse("x;y\n1;2", { delimiter: ";" }), [["x", "y"], ["1", "2"]]);
});

test("hidden: unterminated quote on a later line", () => {
  assert.throws(() => parse('h\r\nok\r\n"open\r\nmore\r\nlines'), (e) => e instanceof CsvError && e.line === 3);
});

test("hidden: empty input", () => {
  assert.deepEqual(parse(""), []);
});
