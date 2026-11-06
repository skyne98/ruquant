import { test } from "node:test";
import assert from "node:assert/strict";
import { parse, parseObjects, stringify } from "../src/index.js";
import { CsvError } from "../src/parse.js";

test("simple records", () => {
  assert.deepEqual(parse("a,b\n1,2\n"), [["a", "b"], ["1", "2"]]);
});

test("quoted delimiter and newline", () => {
  assert.deepEqual(parse('x,"a,b","c\nd"\n'), [["x", "a,b", "c\nd"]]);
});

test("escaped quotes", () => {
  assert.deepEqual(parse('"she said ""hi"""\n'), [['she said "hi"']]);
});

test("CRLF line endings", () => {
  assert.deepEqual(parse("a,b\r\n1,2\r\n"), [["a", "b"], ["1", "2"]]);
});

test("empty trailing field", () => {
  assert.deepEqual(parse("a,b,\n1,2,\n"), [["a", "b", ""], ["1", "2", ""]]);
});

test("unterminated quote reports the line where it starts", () => {
  assert.throws(() => parse('a\n"b\nc'), (e) => e instanceof CsvError && e.line === 2);
});

test("round trip", () => {
  const rows = [["id", "note"], ["1", 'has "quotes", commas\nand lines'], ["2", ""]];
  assert.deepEqual(parse(stringify(rows)), rows);
});

test("objects", () => {
  assert.deepEqual(parseObjects("name,age\nann,31\nbob\n"), [{ name: "ann", age: "31" }, { name: "bob", age: "" }]);
});
