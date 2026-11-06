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
  let quoteLine = 0;
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
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
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
      quoteLine = line;
      i++;
    } else if (c === delimiter) {
      endField();
      i++;
    } else if (c === "\n" || (c === "\r" && text[i + 1] === "\n")) {
      endRecord();
      line++;
      i += c === "\r" ? 2 : 1;
    } else {
      field += c;
      i++;
    }
  }
  if (inQuotes) throw new CsvError("unterminated quoted field", quoteLine);
  if (field !== "" || wasQuoted || record.length) endRecord();
  return records;
}

export function parseObjects(text, options = {}) {
  const [header, ...rows] = parse(text, options);
  if (!header) return [];
  return rows.map((row) => Object.fromEntries(header.map((name, j) => [name, row[j] ?? ""])));
}
