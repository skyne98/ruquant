// RFC 4180 CSV parser. Fields are separated by `delimiter`, records by CRLF or LF. A field may be quoted with `"`;
// inside quotes, delimiters and line breaks are literal and `""` is an escaped quote. A quote inside an unquoted field
// is literal. A trailing line break at the end of the input does not start a new record. An unterminated quoted
// field is an error reported at the line where the field starts.

export class CsvError extends Error {
  constructor(message, line) {
    super(`${message} (line ${line})`);
    this.line = line;
  }
}

export function parse(text, { delimiter = "," } = {}) {
  const records = [];
  let record = [];
  let field = "";
  let inQuotes = false;
  let wasQuoted = false;
  let line = 1;
  let i = 0;

  const endField = () => {
    record.push(field);
    field = "";
    wasQuoted = false;
  };
  const endRecord = () => {
    endField();
    records.push(record);
    record = [];
  };

  while (i < text.length) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        inQuotes = false;
      } else {
        if (c === "\n") line++;
        field += c;
      }
      i++;
      continue;
    }
    if (c === '"' && field === "" && !wasQuoted) {
      inQuotes = true;
      wasQuoted = true;
      i++;
    } else if (c === delimiter) {
      endField();
      i++;
    } else if (c === "\n") {
      endRecord();
      line++;
      i++;
    } else {
      field += c;
      i++;
    }
  }
  if (inQuotes) throw new CsvError("unterminated quoted field", line);
  if (field !== "") endRecord();
  return records;
}

export function parseObjects(text, options = {}) {
  const [header, ...rows] = parse(text, options);
  if (!header) return [];
  return rows.map((row) => Object.fromEntries(header.map((name, j) => [name, row[j] ?? ""])));
}
