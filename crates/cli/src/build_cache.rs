use std::fs;
use std::io::Write;
use std::path::PathBuf;

use sha2::{Digest, Sha256};
use yahtzee_core::Scores;

fn main() -> std::io::Result<()> {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("web/static"));
    fs::create_dir_all(&out_dir)?;

    eprintln!("[build-cache] computing Scores::new() ...");
    let scores = Scores::new();

    eprintln!("[build-cache] serializing with bincode ...");
    let raw: Vec<u8> = bincode::serialize(&scores).expect("serialize");
    let raw_path = out_dir.join("scores.bin");
    fs::write(&raw_path, &raw)?;

    eprintln!("[build-cache] compressing with brotli (q=11, lgwin=24) ...");
    let mut compressed: Vec<u8> = Vec::new();
    {
        let params = brotli::enc::BrotliEncoderParams {
            quality: 11,
            lgwin: 24,
            ..Default::default()
        };
        let mut reader: &[u8] = &raw;
        brotli::BrotliCompress(&mut reader, &mut compressed, &params)?;
    }
    let br_path = out_dir.join("scores.bin.br");
    fs::write(&br_path, &compressed)?;

    let raw_hash = hex(&Sha256::digest(&raw));
    let br_hash = hex(&Sha256::digest(&compressed));
    let manifest = format!(
        "scores.bin    {:>10} bytes  sha256:{}\n\
         scores.bin.br {:>10} bytes  sha256:{}\n",
        raw.len(),
        raw_hash,
        compressed.len(),
        br_hash,
    );
    let manifest_path = out_dir.join("MANIFEST");
    let mut m = fs::File::create(&manifest_path)?;
    m.write_all(manifest.as_bytes())?;

    eprintln!("{}", manifest);
    eprintln!(
        "[build-cache] wrote {}, {}, {}",
        raw_path.display(),
        br_path.display(),
        manifest_path.display()
    );
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
