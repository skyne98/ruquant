//! Minimal GGUF v3 reader / writer: metadata kept as raw bytes (so it round-trips untouched), tensor
//! data read from a memory map, written back one tensor at a time.

use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q8_0: u32 = 8;
pub const BF16: u32 = 30;
pub const W4A4: u32 = 44; // ruquant types, see ggml-common.h block_w4a4
pub const W4A4H: u32 = 45;

/// (block size, bytes per block) of the ggml types we may meet; None for the rest.
pub fn type_layout(t: u32) -> Option<(usize, usize)> {
    Some(match t {
        0 => (1, 4),          // F32
        1 => (1, 2),          // F16
        2 => (32, 18),        // Q4_0
        3 => (32, 20),        // Q4_1
        6 => (32, 22),        // Q5_0
        7 => (32, 24),        // Q5_1
        8 => (32, 34),        // Q8_0
        10 => (256, 84),      // Q2_K
        11 => (256, 110),     // Q3_K
        12 => (256, 144),     // Q4_K
        13 => (256, 176),     // Q5_K
        14 => (256, 210),     // Q6_K
        15 => (256, 292),     // Q8_K
        24 => (1, 1),         // I8
        25 => (1, 2),         // I16
        26 => (1, 4),         // I32
        30 => (1, 2),         // BF16
        44 | 45 => (256, 144), // W4A4, W4A4H
        _ => return None,
    })
}

pub struct Kv {
    pub key: String,
    pub raw: Vec<u8>, // value type (u32) + value, as stored
}

pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>, // ne0 first
    pub ty: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    pub fn n_bytes(&self) -> Result<usize, String> {
        let (blk, size) = type_layout(self.ty).ok_or(format!("{}: unsupported ggml type {}", self.name, self.ty))?;
        Ok(self.n_elements() as usize / blk * size)
    }
}

pub struct Gguf {
    pub map: Mmap,
    pub kvs: Vec<Kv>,
    pub tensors: Vec<TensorInfo>,
    pub data_start: usize,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.p..self.p + n).ok_or("truncated GGUF")?;
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn skip_value(&mut self, ty: u32) -> Result<(), String> {
        match ty {
            0 | 1 | 7 => self.p += 1,
            2 | 3 => self.p += 2,
            4 | 5 | 6 => self.p += 4,
            10 | 11 | 12 => self.p += 8,
            8 => {
                self.string()?;
            }
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.skip_value(et)?;
                }
            }
            _ => return Err(format!("unknown GGUF value type {ty}")),
        }
        Ok(())
    }
}

impl Gguf {
    pub fn open(path: &str) -> Result<Gguf, String> {
        let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let map = unsafe { Mmap::map(&f) }.map_err(|e| format!("{path}: {e}"))?;
        let (kvs, tensors, header_end) = {
            let mut c = Cursor { b: &map, p: 0 };
            if c.take(4)? != b"GGUF" {
                return Err(format!("{path}: not a GGUF file"));
            }
            let version = c.u32()?;
            if version != 3 {
                return Err(format!("{path}: GGUF version {version}, only 3 is supported"));
            }
            let (n_tensors, n_kv) = (c.u64()?, c.u64()?);
            let mut kvs = Vec::new();
            for _ in 0..n_kv {
                let key = c.string()?;
                let start = c.p;
                let ty = c.u32()?;
                c.skip_value(ty)?;
                kvs.push(Kv { key, raw: map[start..c.p].to_vec() });
            }
            let mut tensors = Vec::new();
            for _ in 0..n_tensors {
                let name = c.string()?;
                let nd = c.u32()? as usize;
                let dims = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>, _>>()?;
                let (ty, offset) = (c.u32()?, c.u64()?);
                tensors.push(TensorInfo { name, dims, ty, offset });
            }
            (kvs, tensors, c.p)
        };
        let mut g = Gguf { map, kvs, tensors, data_start: 0 };
        let align = g.alignment();
        g.data_start = header_end.div_ceil(align) * align;
        Ok(g)
    }

    pub fn alignment(&self) -> usize {
        self.u32("general.alignment").unwrap_or(32) as usize
    }

    /// A scalar u32 / i32 / u64 metadata value.
    pub fn u32(&self, key: &str) -> Option<u32> {
        let kv = self.kvs.iter().find(|k| k.key == key)?;
        let ty = u32::from_le_bytes(kv.raw[..4].try_into().unwrap());
        let v = &kv.raw[4..];
        Some(match ty {
            4 | 5 => u32::from_le_bytes(v[..4].try_into().unwrap()),
            10 | 11 => u64::from_le_bytes(v[..8].try_into().unwrap()) as u32,
            _ => return None,
        })
    }

    /// A string metadata value.
    pub fn string(&self, key: &str) -> Option<String> {
        let kv = self.kvs.iter().find(|k| k.key == key)?;
        if u32::from_le_bytes(kv.raw[..4].try_into().unwrap()) != 8 {
            return None;
        }
        let n = u64::from_le_bytes(kv.raw[4..12].try_into().unwrap()) as usize;
        Some(String::from_utf8_lossy(&kv.raw[12..12 + n]).into_owned())
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn data(&self, t: &TensorInfo) -> Result<&[u8], String> {
        let start = self.data_start + t.offset as usize;
        self.map.get(start..start + t.n_bytes()?).ok_or(format!("{}: data out of file", t.name))
    }
}

/// A string metadata value, as raw bytes for a Kv.
pub fn kv_string(key: &str, value: &str) -> Kv {
    let mut raw = 8u32.to_le_bytes().to_vec();
    raw.extend((value.len() as u64).to_le_bytes());
    raw.extend(value.as_bytes());
    Kv { key: key.to_string(), raw }
}

fn put_string(w: &mut impl Write, s: &str) -> std::io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

/// Writes header, metadata and tensor infos (offsets laid out from the given types and dims), then
/// returns a writer positioned at the data section; the caller writes tensors in order with `pad`.
pub struct Writer {
    pub w: BufWriter<File>,
    align: usize,
}

impl Writer {
    pub fn create(path: &str, kvs: &[Kv], tensors: &[TensorInfo], align: usize) -> Result<Writer, String> {
        let f = File::create(path).map_err(|e| format!("{path}: {e}"))?;
        let mut w = BufWriter::with_capacity(64 << 20, f);
        let io = |e: std::io::Error| format!("{path}: {e}");
        w.write_all(b"GGUF").map_err(io)?;
        w.write_all(&3u32.to_le_bytes()).map_err(io)?;
        w.write_all(&(tensors.len() as u64).to_le_bytes()).map_err(io)?;
        w.write_all(&(kvs.len() as u64).to_le_bytes()).map_err(io)?;
        for kv in kvs {
            put_string(&mut w, &kv.key).map_err(io)?;
            w.write_all(&kv.raw).map_err(io)?;
        }
        let mut offset = 0u64;
        for t in tensors {
            put_string(&mut w, &t.name).map_err(io)?;
            w.write_all(&(t.dims.len() as u32).to_le_bytes()).map_err(io)?;
            for d in &t.dims {
                w.write_all(&d.to_le_bytes()).map_err(io)?;
            }
            w.write_all(&t.ty.to_le_bytes()).map_err(io)?;
            w.write_all(&offset.to_le_bytes()).map_err(io)?;
            offset += (t.n_bytes()?.div_ceil(align) * align) as u64;
        }
        let mut wr = Writer { w, align };
        wr.pad().map_err(io)?;
        Ok(wr)
    }

    /// Writes one tensor's data followed by padding to the alignment.
    pub fn tensor(&mut self, data: &[u8]) -> Result<(), String> {
        self.w.write_all(data).map_err(|e| e.to_string())?;
        self.pad().map_err(|e| e.to_string())
    }

    fn pad(&mut self) -> std::io::Result<()> {
        let pos = self.w.stream_position()? as usize;
        let n = pos.div_ceil(self.align) * self.align - pos;
        self.w.write_all(&vec![0u8; n])
    }

    pub fn finish(mut self) -> Result<(), String> {
        self.w.flush().map_err(|e| e.to_string())
    }
}
