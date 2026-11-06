export function stringify(records, { delimiter = ",", eol = "\r\n" } = {}) {
  const quote = (v) => {
    const s = v == null ? "" : String(v);
    return /[",\r\n]/.test(s) || s.includes(delimiter) ? `"${s.replaceAll('"', '""')}"` : s;
  };
  return records.map((r) => r.map(quote).join(delimiter)).join(eol) + (records.length ? eol : "");
}
