//! `easy-yahtzee` — CLI for the Yahtzee solver.
//!
//! Subcommands:
//!   * `solve`  — recommend the next move from a position.
//!   * `value`  — print the overall expected final score from a position.
//!   * `build`  — generate the precomputed score table (~10s on a modern
//!                desktop at release; minutes on slower hardware or CI).
//!
//! `solve` and `value` use the score table embedded in the binary by default
//! (`crates/cli/data/scores.bin.br`, brotli-decompressed at startup).
//! `--scores PATH` overrides the embedded blob with a raw bincode file from
//! disk — useful for testing a freshly-rebuilt table before checking it in.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use sha2::{Digest, Sha256};

use yahtzee_core::{recommend, Recommendation, Scores, StateInput};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "easy-yahtzee",
    version,
    about = "Optimal Yahtzee solver",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Recommend the next move from a position.
    Solve(SolveArgs),
    /// Print the overall expected final score from a position.
    Value(ValueArgs),
    /// Generate the precomputed score table and write it to disk. ~10s at
    /// release on a modern desktop, longer at debug or on slower hardware.
    /// Typically run once per `crates/core` serialization-layout change.
    Build(BuildArgs),
}

/// Args shared by `solve` and `value`: everything that locates a state in
/// scorecard-space, plus the score-table source.
#[derive(Args, Clone)]
struct StateArgs {
    /// Override the embedded score table with a raw bincode file produced by
    /// `easy-yahtzee build` (no `--brotli`). Useful when iterating on
    /// `crates/core` without rebuilding the CLI binary.
    #[arg(long, value_name = "PATH")]
    scores: Option<PathBuf>,

    /// Already-filled categories, comma-separated. Names: ones, twos, threes,
    /// fours, fives, sixes, three-of-a-kind, four-of-a-kind, full-house,
    /// small-straight, large-straight, yahtzee, chance. Short aliases also
    /// accepted: 1s..6s, 3k, 4k, fh, ss, ls, y, ch. Example:
    /// `--filled ones,full-house,chance`.
    #[arg(short = 'f', long, value_parser = parse_filled, default_value = "")]
    filled: [bool; 13],

    /// Upper-section points still needed to clear the +35 bonus (0..=63).
    /// Default 63 means "fresh game, no upper categories filled yet".
    #[arg(
        short = 'u',
        long,
        default_value_t = 63,
        value_parser = clap::value_parser!(u8).range(0..=63),
    )]
    upper_remaining: u8,

    /// Set if a Yahtzee has already been scored (as a real Yahtzee, not a
    /// zero) earlier in the game, so any further Yahtzees award the +100
    /// bonus and are eligible for the joker rule.
    #[arg(short = 'y', long)]
    yahtzee_bonus_eligible: bool,
}

#[derive(Args)]
struct SolveArgs {
    #[command(flatten)]
    state: StateArgs,

    /// Which roll of the turn (1, 2, or 3).
    #[arg(short = 'r', long, value_parser = clap::value_parser!(u8).range(1..=3))]
    roll: u8,

    /// Five dice faces. Either comma-separated (`1,1,3,4,6`) or packed
    /// (`11346`).
    #[arg(short = 'd', long, value_parser = parse_dice)]
    dice: [u8; 5],

    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,
}

#[derive(Args)]
struct ValueArgs {
    #[command(flatten)]
    state: StateArgs,

    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,
}

#[derive(Args)]
struct BuildArgs {
    /// Path for the bincode-serialized score table.
    #[arg(short = 'o', long, default_value = "scores.bin")]
    output: PathBuf,

    /// Also write `<output>.br` (brotli q=11, lgwin=24) and a `MANIFEST` with
    /// SHA-256 hashes alongside `<output>`. This is the bundle the web
    /// frontend consumes.
    #[arg(long)]
    brotli: bool,
}

#[derive(Copy, Clone, ValueEnum)]
enum Format {
    Text,
    Json,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Solve(args) => cmd_solve(args),
        Cmd::Value(args) => cmd_value(args),
        Cmd::Build(args) => cmd_build(args),
    }
}

fn cmd_solve(args: SolveArgs) -> Result<()> {
    let scores = load_or_embedded_scores(args.state.scores.as_deref())?;
    let input = state_input(&args.state);
    let rec = recommend(&scores, &input, &args.dice, args.roll)
        .map_err(|e| anyhow!("recommend failed: {e}"))?;

    match args.format {
        Format::Text => render_solve_text(&rec, args.roll),
        Format::Json => render_json(&rec)?,
    }
    Ok(())
}

fn cmd_value(args: ValueArgs) -> Result<()> {
    let scores = load_or_embedded_scores(args.state.scores.as_deref())?;
    let input = state_input(&args.state);
    let state = yahtzee_core::build_state(&input).map_err(|e| anyhow!(e))?;
    let value = scores.values(state).value;

    match args.format {
        Format::Text => println!("{value:.4}"),
        Format::Json => {
            let body = serde_json::json!({ "value": value });
            println!("{}", serde_json::to_string(&body)?);
        }
    }
    Ok(())
}

fn cmd_build(args: BuildArgs) -> Result<()> {
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }

    eprintln!("[build] computing Scores::new() ...");
    let scores = Scores::new();

    eprintln!("[build] serializing with bincode ...");
    let raw: Vec<u8> = bincode::serialize(&scores).context("bincode serialize")?;
    fs::write(&args.output, &raw)
        .with_context(|| format!("writing {}", args.output.display()))?;
    eprintln!("[build] wrote {} ({} bytes)", args.output.display(), raw.len());

    if args.brotli {
        let br_path = sibling_with_extension(&args.output, "br");
        let manifest_path = args
            .output
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("MANIFEST");

        eprintln!("[build] compressing with brotli (q=11, lgwin=24) ...");
        let mut compressed: Vec<u8> = Vec::new();
        let params = brotli::enc::BrotliEncoderParams {
            quality: 11,
            lgwin: 24,
            ..Default::default()
        };
        let mut reader: &[u8] = &raw;
        brotli::BrotliCompress(&mut reader, &mut compressed, &params)
            .context("brotli compress")?;
        fs::write(&br_path, &compressed)
            .with_context(|| format!("writing {}", br_path.display()))?;

        let raw_hash = hex(&Sha256::digest(&raw));
        let br_hash = hex(&Sha256::digest(&compressed));
        let bin_name = args
            .output
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "scores.bin".to_string());
        let br_name = br_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "scores.bin.br".to_string());
        let manifest = format!(
            "{bin_name:<13} {:>10} bytes  sha256:{raw_hash}\n\
             {br_name:<13} {:>10} bytes  sha256:{br_hash}\n",
            raw.len(),
            compressed.len(),
        );
        let mut m = fs::File::create(&manifest_path)
            .with_context(|| format!("writing {}", manifest_path.display()))?;
        m.write_all(manifest.as_bytes())?;

        eprintln!("{manifest}");
        eprintln!(
            "[build] wrote {}, {}, {}",
            args.output.display(),
            br_path.display(),
            manifest_path.display()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers: scores I/O, state construction
// ---------------------------------------------------------------------------

/// The canonical solver table, brotli-compressed bincode of `Scores`. Tracked
/// in git at `crates/cli/data/scores.bin.br` so fresh clones can build the CLI
/// without first running the multi-minute `Scores::new()` from `crates/core`.
const EMBEDDED_SCORES_BR: &[u8] = include_bytes!("../data/scores.bin.br");

fn load_or_embedded_scores(path: Option<&std::path::Path>) -> Result<Scores> {
    match path {
        Some(p) => load_scores(p),
        None => embedded_scores(),
    }
}

fn load_scores(path: &std::path::Path) -> Result<Scores> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let scores: Scores =
        bincode::deserialize(&bytes).with_context(|| format!("deserializing {}", path.display()))?;
    Ok(scores)
}

/// Decompress and deserialize the embedded score table. Allocates ~4 MiB for
/// the decompressed bincode buffer; the decode itself takes tens of ms.
fn embedded_scores() -> Result<Scores> {
    let mut decompressed: Vec<u8> = Vec::with_capacity(4 * 1024 * 1024);
    let mut reader: &[u8] = EMBEDDED_SCORES_BR;
    brotli::BrotliDecompress(&mut reader, &mut decompressed)
        .context("decompressing embedded score table")?;
    let scores: Scores = bincode::deserialize(&decompressed)
        .context("deserializing embedded score table (regenerate the embed?)")?;
    Ok(scores)
}

fn state_input(args: &StateArgs) -> StateInput {
    StateInput {
        entries: args.filled,
        yahtzee_bonus_eligible: args.yahtzee_bonus_eligible,
        upper_score_remaining: args.upper_remaining,
    }
}

fn sibling_with_extension(path: &std::path::Path, added_ext: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".");
    s.push(added_ext);
    PathBuf::from(s)
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

// ---------------------------------------------------------------------------
// Parsers
// ---------------------------------------------------------------------------

/// Parse `--dice` from either `1,1,3,4,6` or `11346`.
fn parse_dice(s: &str) -> Result<[u8; 5], String> {
    let digits: Vec<u8> = if s.contains(',') {
        s.split(',')
            .map(|t| {
                t.trim()
                    .parse::<u8>()
                    .map_err(|e| format!("bad die `{t}`: {e}"))
            })
            .collect::<Result<_, _>>()?
    } else {
        s.chars()
            .map(|c| {
                c.to_digit(10)
                    .map(|d| d as u8)
                    .ok_or_else(|| format!("bad die `{c}` (expected 1..6)"))
            })
            .collect::<Result<_, _>>()?
    };

    if digits.len() != 5 {
        return Err(format!("expected 5 dice, got {}", digits.len()));
    }
    for &d in &digits {
        if !(1..=6).contains(&d) {
            return Err(format!("die out of range: {d} (expected 1..=6)"));
        }
    }
    Ok([digits[0], digits[1], digits[2], digits[3], digits[4]])
}

/// Parse `--filled` from a comma-separated list of category names.
/// Empty input → no categories filled.
fn parse_filled(s: &str) -> Result<[bool; 13], String> {
    let mut out = [false; 13];
    let s = s.trim();
    if s.is_empty() {
        return Ok(out);
    }
    for tok in s.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        let idx = category_index(t)
            .ok_or_else(|| format!("unknown category `{t}` (try ones..sixes, full-house, …)"))?;
        if out[idx] {
            return Err(format!("category `{t}` listed twice"));
        }
        out[idx] = true;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Category names
// ---------------------------------------------------------------------------

/// Aligned with `yahtzee_core::ENTRY_ACTIONS`.
const CATEGORY_NAMES: [&str; 13] = [
    "ones",
    "twos",
    "threes",
    "fours",
    "fives",
    "sixes",
    "three-of-a-kind",
    "four-of-a-kind",
    "full-house",
    "small-straight",
    "large-straight",
    "yahtzee",
    "chance",
];

/// `(alias, canonical_index)` pairs. Canonical names from `CATEGORY_NAMES` are
/// always accepted; this table is only the *short* aliases.
const CATEGORY_ALIASES: &[(&str, usize)] = &[
    ("1s", 0),
    ("2s", 1),
    ("3s", 2),
    ("4s", 3),
    ("5s", 4),
    ("6s", 5),
    ("3k", 6),
    ("4k", 7),
    ("fh", 8),
    ("ss", 9),
    ("ls", 10),
    ("y", 11),
    ("ch", 12),
];

fn category_index(name: &str) -> Option<usize> {
    let n = name.to_ascii_lowercase();
    if let Some(i) = CATEGORY_NAMES.iter().position(|c| *c == n) {
        return Some(i);
    }
    CATEGORY_ALIASES
        .iter()
        .find_map(|&(alias, idx)| (alias == n).then_some(idx))
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------

/// Merged choice: either "keep these dice" or "score in this category".
struct Choice {
    label: String,
    ev: f32,
    turn_ev: f32,
}

fn render_solve_text(rec: &Recommendation, roll: u8) {
    println!("state EV: {:.4}", rec.value);
    println!();

    let mut choices: Vec<Choice> = Vec::new();
    if let Some(keepers) = &rec.keepers {
        for k in keepers {
            choices.push(Choice {
                label: format!("keep {}", format_dice(&k.dice)),
                ev: k.ev,
                turn_ev: k.turn_ev,
            });
        }
    }
    for e in &rec.entries {
        let prefix = if roll == 3 { "score" } else { "score now in" };
        choices.push(Choice {
            label: format!("{prefix} {}", CATEGORY_NAMES[e.entry as usize]),
            ev: e.ev,
            turn_ev: e.turn_ev,
        });
    }
    // Stable sort by overall EV descending. Ties keep keepers-vs-entries
    // ordering as inserted (keepers first), matching the web UI.
    choices.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap_or(std::cmp::Ordering::Equal));

    let label_width = choices.iter().map(|c| c.label.len()).max().unwrap_or(0);
    println!(
        "  {:<label_width$}    overall EV    this turn",
        "choice",
        label_width = label_width.max("choice".len()),
    );
    for c in &choices {
        println!(
            "  {:<label_width$}    {:>10.4}    {:>9.4}",
            c.label,
            c.ev,
            c.turn_ev,
            label_width = label_width.max("choice".len()),
        );
    }
}

fn render_json(rec: &Recommendation) -> Result<()> {
    // Augment each entry with its category name so JSON consumers don't have
    // to maintain their own index → name table.
    let entries: Vec<serde_json::Value> = rec
        .entries
        .iter()
        .map(|e| {
            serde_json::json!({
                "entry": e.entry,
                "name": CATEGORY_NAMES[e.entry as usize],
                "ev": e.ev,
                "turn_ev": e.turn_ev,
            })
        })
        .collect();
    let body = serde_json::json!({
        "value": rec.value,
        "keepers": rec.keepers,
        "entries": entries,
    });
    println!("{}", serde_json::to_string(&body)?);
    Ok(())
}

fn format_dice(dice: &[u8]) -> String {
    if dice.is_empty() {
        return "(nothing)".to_string();
    }
    dice.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dice_comma() {
        assert_eq!(parse_dice("1,2,3,4,5").unwrap(), [1, 2, 3, 4, 5]);
        assert_eq!(parse_dice("6,6,6,6,6").unwrap(), [6, 6, 6, 6, 6]);
    }

    #[test]
    fn parse_dice_packed() {
        assert_eq!(parse_dice("12345").unwrap(), [1, 2, 3, 4, 5]);
        assert_eq!(parse_dice("11346").unwrap(), [1, 1, 3, 4, 6]);
    }

    #[test]
    fn parse_dice_rejects_bad_input() {
        assert!(parse_dice("1,2,3,4").is_err()); // too short
        assert!(parse_dice("1234").is_err()); // too short
        assert!(parse_dice("123456").is_err()); // too long
        assert!(parse_dice("1,2,3,4,7").is_err()); // out of range
        assert!(parse_dice("12340").is_err()); // out of range (0)
        assert!(parse_dice("1234x").is_err()); // non-digit
    }

    #[test]
    fn parse_filled_empty() {
        assert_eq!(parse_filled("").unwrap(), [false; 13]);
        assert_eq!(parse_filled("   ").unwrap(), [false; 13]);
    }

    #[test]
    fn parse_filled_canonical_and_aliases() {
        let f = parse_filled("ones,full-house,chance").unwrap();
        let mut want = [false; 13];
        want[0] = true; // ones
        want[8] = true; // full-house
        want[12] = true; // chance
        assert_eq!(f, want);

        // Aliases match the same indices as canonical names.
        assert_eq!(parse_filled("1s,fh,ch").unwrap(), want);
    }

    #[test]
    fn parse_filled_rejects_unknown_and_dupes() {
        assert!(parse_filled("ones,banana").is_err());
        assert!(parse_filled("ones,1s").is_err()); // same category twice
    }
}
