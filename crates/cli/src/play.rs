//! Interactive `play` subcommand: drive a full Yahtzee game with the solver
//! as a coach (default), as an autopilot (`--auto`), or against real dice
//! you're rolling at the table (`--manual-dice`).
//!
//! The game state lives in [`Scorecard`] (a thin wrapper over a per-category
//! `Option<u8>` plus Yahtzee-bonus bookkeeping). Every turn we:
//!   1. Roll (or prompt for) five dice.
//!   2. Up to three rolls: ask the solver via [`Scores::recommend`], display
//!      the top pick + alternatives, and accept a user (or auto) action.
//!   3. When the action is "score", apply it to the scorecard and end the
//!      turn. Bonuses (+35 upper, +100 yahtzee) are computed locally — no
//!      need to round-trip through `score_and_child`.
//!
//! The auto+seeded path is fully deterministic and is exercised by
//! `tests/play_auto.rs` to keep the loop, scoring, and bonus logic honest.

use std::io::{self, BufRead, IsTerminal, Write};

use anyhow::{anyhow, bail, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};

use yahtzee_core::{DiceCounts, EntryAction, Recommendation, Scores, State, ENTRY_ACTIONS};

use crate::{category_index, format_dice, parse_dice, PlayArgs, CATEGORY_NAMES};

const YAHTZEE_IDX: usize = 11;
const UPPER_BONUS_THRESHOLD: u32 = 63;
const UPPER_BONUS: u32 = 35;
const YAHTZEE_BONUS: u32 = 100;

/// Display labels for each category, aligned with `CATEGORY_NAMES` /
/// `ENTRY_ACTIONS`. The kebab-case names parse on input but read poorly in a
/// rendered scorecard, so the box uses these instead.
const DISPLAY_NAMES: [&str; 13] = [
    "Ones",
    "Twos",
    "Threes",
    "Fours",
    "Fives",
    "Sixes",
    "Three of a Kind",
    "Four of a Kind",
    "Full House",
    "Small Straight",
    "Large Straight",
    "Yahtzee",
    "Chance",
];

// Box-rendering geometry. Held as constants so the totals padding stays in
// lockstep with the per-category cells without anyone needing to count chars.
const LABEL_W: usize = 16; // width of "Three of a Kind"
const VALUE_W: usize = 3; // widest value is "50" (Yahtzee), so 2 → 3 with padding
const CELL_W: usize = 2 + LABEL_W + 2 + VALUE_W + 2; // 25
const INNER_W: usize = CELL_W * 2; // 50

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------

/// Tiny ANSI styler. We intentionally hand-roll the codes instead of pulling
/// in `crossterm` — the surface area we use is so small (dim / green / bold)
/// that the dep cost outweighs the convenience. `enabled` is set from
/// `IsTerminal::is_terminal()` so piped output stays plain.
pub(crate) struct Style {
    pub enabled: bool,
}

impl Style {
    pub fn detect() -> Self {
        Self {
            enabled: io::stdout().is_terminal(),
        }
    }

    fn wrap(&self, code: &str, s: &str) -> String {
        if self.enabled {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn dim(&self, s: &str) -> String {
        self.wrap("2", s)
    }
    pub fn green(&self, s: &str) -> String {
        self.wrap("32", s)
    }
    pub fn bold(&self, s: &str) -> String {
        self.wrap("1", s)
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run(args: PlayArgs, scores: Scores) -> Result<()> {
    let style = Style::detect();
    run_with_io(
        args,
        scores,
        &mut io::stdin().lock(),
        &mut io::stdout(),
        &style,
    )?;
    Ok(())
}

/// Inner entry point that takes injected I/O and an explicit `Style`. The
/// integration test goes through `run` (no colors because piped stdout isn't
/// a TTY); unit tests can construct a `Style { enabled: false }` directly to
/// keep their golden output simple.
pub(crate) fn run_with_io<R: BufRead, W: Write>(
    args: PlayArgs,
    scores: Scores,
    input: &mut R,
    output: &mut W,
    style: &Style,
) -> Result<u32> {
    let mut rng = match args.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };
    let mut card = Scorecard::default();

    let opening_ev = scores.values(card.state()).value;
    writeln!(output, "=== Easy Yahtzee ===")?;
    writeln!(
        output,
        "Optimal expected score from a fresh game: {opening_ev:.2}",
    )?;
    if args.auto {
        writeln!(output, "Mode: auto-play (solver picks every action).")?;
    }
    if args.manual_dice {
        writeln!(output, "Mode: manual dice entry.")?;
    }
    writeln!(output)?;

    let mut turn = 1u32;
    while !card.is_complete() {
        play_turn(&mut card, &scores, &mut rng, &args, turn, input, output, style)?;
        turn += 1;
        writeln!(output)?;
    }

    writeln!(output, "=== Game over ===")?;
    print_card(&card, style, output)?;
    let achieved = card.grand_total();
    let delta = achieved as f32 - opening_ev;
    writeln!(output)?;
    writeln!(output, "Final score:    {achieved}")?;
    writeln!(output, "Expected (EV):  {opening_ev:.2}")?;
    if delta >= 0.0 {
        writeln!(output, "Above expectation by {:.2}", delta)?;
    } else {
        writeln!(output, "Below expectation by {:.2} (variance + suboptimal play)", -delta)?;
    }
    Ok(achieved)
}

// ---------------------------------------------------------------------------
// Scorecard
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Debug)]
pub(crate) struct Scorecard {
    /// Filled entries. `None` = empty box, `Some(v)` = scored `v` (0..=50).
    pub entries: [Option<u8>; 13],
    /// True iff the YAHTZEE box was filled with an actual yahtzee (50 pts).
    /// Equivalent to `State::yahtzee_bonus_eligible`.
    pub yahtzee_was_real: bool,
    /// Number of +100 Yahtzee bonuses earned (each subsequent yahtzee after
    /// the first one in the YAHTZEE box, if that one scored 50).
    pub yahtzee_bonus_count: u32,
}

impl Scorecard {
    pub fn state(&self) -> State {
        let mut bits = EntryAction::empty();
        for (i, e) in self.entries.iter().enumerate() {
            if e.is_some() {
                bits |= ENTRY_ACTIONS[i];
            }
        }
        let upper_filled = self.upper_total();
        let upper_score_remaining =
            UPPER_BONUS_THRESHOLD.saturating_sub(upper_filled.min(UPPER_BONUS_THRESHOLD)) as u8;
        State {
            entries: bits,
            yahtzee_bonus_eligible: self.yahtzee_was_real,
            upper_score_remaining,
        }
    }

    pub fn upper_total(&self) -> u32 {
        (0..6).map(|i| self.entries[i].unwrap_or(0) as u32).sum()
    }

    pub fn lower_total(&self) -> u32 {
        (6..13).map(|i| self.entries[i].unwrap_or(0) as u32).sum()
    }

    pub fn upper_bonus(&self) -> u32 {
        if self.upper_total() >= UPPER_BONUS_THRESHOLD {
            UPPER_BONUS
        } else {
            0
        }
    }

    pub fn yahtzee_bonus_total(&self) -> u32 {
        self.yahtzee_bonus_count * YAHTZEE_BONUS
    }

    pub fn grand_total(&self) -> u32 {
        self.upper_total() + self.upper_bonus() + self.lower_total() + self.yahtzee_bonus_total()
    }

    pub fn is_complete(&self) -> bool {
        self.entries.iter().all(|e| e.is_some())
    }

    /// Apply a scoring decision: write the entry, update bonus tracking,
    /// return the breakdown for display.
    pub fn apply(&mut self, entry_idx: usize, dice: &DiceCounts) -> ApplyResult {
        let state = self.state();
        let was_real_before = state.yahtzee_bonus_eligible;
        let upper_before = self.upper_total();

        let action = ENTRY_ACTIONS[entry_idx];
        // `entry_score` handles the joker rule for us.
        let entry_value = state.entry_score(action, dice);
        self.entries[entry_idx] = Some(entry_value);

        let dice_is_yahtzee = dice.0.iter().any(|&c| c == 5);

        // Yahtzee bonus: triggers when YAHTZEE was previously filled with a
        // real 50, AND this turn's dice are a yahtzee, AND we're scoring in a
        // *different* category (writing into the already-full YAHTZEE box
        // isn't legal anyway, but the != check keeps the logic robust).
        let yahtzee_bonus = if was_real_before && dice_is_yahtzee && entry_idx != YAHTZEE_IDX {
            self.yahtzee_bonus_count += 1;
            YAHTZEE_BONUS
        } else {
            0
        };

        // After the bonus check (so was_real_before is read first), update
        // eligibility for future turns.
        if entry_idx == YAHTZEE_IDX && dice_is_yahtzee {
            self.yahtzee_was_real = true;
        }

        let upper_after = self.upper_total();
        let upper_bonus = if upper_before < UPPER_BONUS_THRESHOLD
            && upper_after >= UPPER_BONUS_THRESHOLD
        {
            UPPER_BONUS
        } else {
            0
        };

        ApplyResult {
            entry_value,
            upper_bonus,
            yahtzee_bonus,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ApplyResult {
    pub entry_value: u8,
    pub upper_bonus: u32,
    pub yahtzee_bonus: u32,
}

// ---------------------------------------------------------------------------
// Choice merging (keeper rows + entry rows by overall EV)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Choice {
    label: String,
    ev: f32,
    turn_ev: f32,
    kind: ChoiceKind,
}

#[derive(Clone)]
enum ChoiceKind {
    Keep(Vec<u8>),
    Score(usize),
}

fn merged_choices(rec: &Recommendation) -> Vec<Choice> {
    let mut out = Vec::new();
    if let Some(keepers) = &rec.keepers {
        for k in keepers {
            out.push(Choice {
                label: format!("keep {}", format_dice(&k.dice)),
                ev: k.ev,
                turn_ev: k.turn_ev,
                kind: ChoiceKind::Keep(k.dice.clone()),
            });
        }
    }
    for e in &rec.entries {
        out.push(Choice {
            label: format!("score {}", CATEGORY_NAMES[e.entry as usize]),
            ev: e.ev,
            turn_ev: e.turn_ev,
            kind: ChoiceKind::Score(e.entry as usize),
        });
    }
    out.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap_or(std::cmp::Ordering::Equal));
    out
}

// ---------------------------------------------------------------------------
// Turn loop
// ---------------------------------------------------------------------------

fn play_turn<R: BufRead, W: Write>(
    card: &mut Scorecard,
    scores: &Scores,
    rng: &mut StdRng,
    args: &PlayArgs,
    turn: u32,
    input: &mut R,
    output: &mut W,
    style: &Style,
) -> Result<()> {
    print_card(card, style, output)?;
    writeln!(output, "--- Turn {turn} ---")?;

    let mut dice: [u8; 5] = if args.manual_dice {
        prompt_dice("Roll 1 dice: ", input, output)?
    } else {
        roll_five(rng)
    };

    for roll in 1..=3u8 {
        let dice_counts = dice_to_counts_arr(&dice);
        let rec = scores
            .recommend(card.state(), dice_counts.clone(), roll)
            .map_err(|e| anyhow!("recommend: {e}"))?;

        let choices = merged_choices(&rec);
        let top = choices.first().ok_or_else(|| anyhow!("no choices"))?;

        writeln!(output)?;
        writeln!(output, "Roll {roll}: {}", format_dice(&dice))?;
        writeln!(
            output,
            "Solver suggests: {} (overall EV {:.2}, turn EV {:.2})",
            top.label, top.ev, top.turn_ev,
        )?;
        let alt_count = args.top.saturating_sub(1).min(choices.len().saturating_sub(1));
        for c in choices.iter().skip(1).take(alt_count) {
            writeln!(
                output,
                "  alt: {} (overall EV {:.2}, turn EV {:.2})",
                c.label, c.ev, c.turn_ev,
            )?;
        }

        let action = if args.auto {
            choice_to_action(top)
        } else {
            prompt_action(roll, &rec, &dice, &choices, card, style, input, output)?
        };

        match action {
            Action::Score(idx) => {
                let result = card.apply(idx, &dice_counts);
                let mut total = result.entry_value as u32;
                let mut detail = format!(
                    "Scored {} in {}",
                    result.entry_value, CATEGORY_NAMES[idx]
                );
                if result.upper_bonus > 0 {
                    total += result.upper_bonus;
                    detail.push_str(&format!(" + {} upper bonus", result.upper_bonus));
                }
                if result.yahtzee_bonus > 0 {
                    total += result.yahtzee_bonus;
                    detail.push_str(&format!(" + {} Yahtzee bonus", result.yahtzee_bonus));
                }
                writeln!(output, "=> {detail} = {total} this turn")?;
                return Ok(());
            }
            Action::Keep(kept) => {
                if roll == 3 {
                    bail!("internal: keep on roll 3 should be impossible");
                }
                if args.manual_dice {
                    let to_roll = 5 - kept.len();
                    if to_roll == 0 {
                        bail!("kept all 5 dice — score them with `s <category>`");
                    }
                    writeln!(
                        output,
                        "Keeping {}; please roll {to_roll} dice and enter them.",
                        format_dice(&kept),
                    )?;
                    let mut combined = kept.clone();
                    combined.extend(prompt_partial(to_roll, input, output)?);
                    combined.sort();
                    dice = combined.as_slice().try_into().unwrap();
                } else {
                    dice = reroll(rng, &kept);
                }
            }
        }
    }

    // Loop body always returns or rerolls; the only way to fall out is roll==3
    // with a Keep action, which we reject above.
    unreachable!("turn ended without scoring")
}

// ---------------------------------------------------------------------------
// Action / prompts
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Action {
    Score(usize),
    Keep(Vec<u8>),
}

fn choice_to_action(c: &Choice) -> Action {
    match &c.kind {
        ChoiceKind::Keep(d) => Action::Keep(d.clone()),
        ChoiceKind::Score(i) => Action::Score(*i),
    }
}

fn prompt_action<R: BufRead, W: Write>(
    roll: u8,
    rec: &Recommendation,
    dice: &[u8; 5],
    choices: &[Choice],
    card: &Scorecard,
    style: &Style,
    input: &mut R,
    output: &mut W,
) -> Result<Action> {
    loop {
        write!(output, "> ")?;
        output.flush()?;
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            bail!("unexpected EOF on input");
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut tokens = line.split_whitespace();
        let cmd = tokens.next().unwrap();
        match cmd {
            "k" | "keep" => {
                if roll == 3 {
                    writeln!(output, "no rolls left — score with `s <category>`.")?;
                    continue;
                }
                let parsed: Result<Vec<u8>, _> = tokens.map(|t| t.parse::<u8>()).collect();
                match parsed {
                    Ok(faces) if faces.iter().all(|&f| (1..=6).contains(&f)) => {
                        if !is_dice_subset(&faces, dice) {
                            writeln!(
                                output,
                                "those dice aren't all in your roll ({} not a subset of {}).",
                                format_dice(&faces),
                                format_dice(dice),
                            )?;
                            continue;
                        }
                        if faces.len() == 5 {
                            writeln!(
                                output,
                                "keeping all 5 means scoring now — use `s <category>`.",
                            )?;
                            continue;
                        }
                        return Ok(Action::Keep(faces));
                    }
                    _ => writeln!(output, "usage: k <face> <face> ... (each 1..=6)")?,
                }
            }
            "s" | "score" => {
                let cat = tokens.collect::<Vec<_>>().join(" ");
                if cat.is_empty() {
                    writeln!(output, "usage: s <category>")?;
                    continue;
                }
                match category_index(&cat) {
                    Some(i) => {
                        // Verify the category is actually open by checking
                        // whether `recommend` exposed it as a scoring option.
                        let open = rec.entries.iter().any(|e| e.entry as usize == i);
                        if !open {
                            writeln!(output, "category `{}` is already filled.", CATEGORY_NAMES[i])?;
                            continue;
                        }
                        return Ok(Action::Score(i));
                    }
                    None => writeln!(output, "unknown category `{cat}`. (try `?` or `h`.)")?,
                }
            }
            "a" | "auto" => {
                return Ok(choice_to_action(&choices[0]));
            }
            "?" | "top" => {
                let n = choices.len().min(8);
                for (i, c) in choices.iter().take(n).enumerate() {
                    writeln!(
                        output,
                        "  {}. {:<28} ev {:>8.2}  turn {:>7.2}",
                        i + 1,
                        c.label,
                        c.ev,
                        c.turn_ev,
                    )?;
                }
            }
            "c" | "card" => print_card(card, style, output)?,
            "h" | "help" => print_help(roll, output)?,
            "q" | "quit" | "exit" => bail!("user quit"),
            _ => print_help(roll, output)?,
        }
    }
}

fn print_help<W: Write>(roll: u8, output: &mut W) -> Result<()> {
    writeln!(output, "Commands:")?;
    if roll < 3 {
        writeln!(output, "  k <faces>     keep these dice (e.g. `k 4 4 6`)")?;
    }
    writeln!(output, "  s <category>  score current dice (e.g. `s fours` / `s fh`)")?;
    writeln!(output, "  a             accept solver's top recommendation")?;
    writeln!(output, "  ?             list top alternatives by EV")?;
    writeln!(output, "  c             redisplay the scorecard")?;
    writeln!(output, "  h             this help")?;
    writeln!(output, "  q             quit")?;
    Ok(())
}

fn prompt_dice<R: BufRead, W: Write>(label: &str, input: &mut R, output: &mut W) -> Result<[u8; 5]> {
    loop {
        write!(output, "{label}")?;
        output.flush()?;
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            bail!("unexpected EOF on input");
        }
        match parse_dice(line.trim()) {
            Ok(d) => return Ok(d),
            Err(e) => writeln!(output, "error: {e}")?,
        }
    }
}

fn prompt_partial<R: BufRead, W: Write>(
    n: usize,
    input: &mut R,
    output: &mut W,
) -> Result<Vec<u8>> {
    loop {
        write!(output, "New {n} dice: ")?;
        output.flush()?;
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            bail!("unexpected EOF on input");
        }
        let line = line.trim();
        let parsed: std::result::Result<Vec<u8>, String> = if line.contains(',') {
            line.split(',')
                .map(|t| t.trim().parse::<u8>().map_err(|e| e.to_string()))
                .collect()
        } else {
            line.chars()
                .map(|c| {
                    c.to_digit(10)
                        .map(|d| d as u8)
                        .ok_or_else(|| format!("bad digit `{c}`"))
                })
                .collect()
        };
        match parsed {
            Ok(p) if p.len() == n && p.iter().all(|&d| (1..=6).contains(&d)) => return Ok(p),
            Ok(p) => writeln!(output, "expected {n} dice in 1..=6, got {p:?}")?,
            Err(e) => writeln!(output, "error: {e}")?,
        }
    }
}

// ---------------------------------------------------------------------------
// Dice helpers
// ---------------------------------------------------------------------------

fn roll_five(rng: &mut StdRng) -> [u8; 5] {
    let mut out = [0u8; 5];
    for slot in &mut out {
        *slot = rng.random_range(1..=6);
    }
    out.sort();
    out
}

fn reroll(rng: &mut StdRng, kept: &[u8]) -> [u8; 5] {
    let mut out = [0u8; 5];
    for (i, &d) in kept.iter().enumerate() {
        out[i] = d;
    }
    for i in kept.len()..5 {
        out[i] = rng.random_range(1..=6);
    }
    out.sort();
    out
}

fn dice_to_counts_arr(dice: &[u8; 5]) -> DiceCounts {
    let mut counts = [0u8; 6];
    for &d in dice {
        counts[(d - 1) as usize] += 1;
    }
    DiceCounts(counts)
}

fn is_dice_subset(needles: &[u8], haystack: &[u8]) -> bool {
    let mut hay = [0u8; 6];
    for &d in haystack {
        if (1..=6).contains(&d) {
            hay[(d - 1) as usize] += 1;
        }
    }
    let mut need = [0u8; 6];
    for &d in needles {
        if !(1..=6).contains(&d) {
            return false;
        }
        need[(d - 1) as usize] += 1;
    }
    (0..6).all(|i| need[i] <= hay[i])
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Boxed scorecard
// ---------------------------------------------------------------------------
//
// Layout (INNER_W = 50):
//
// ┌─ Scorecard ──────────────────────────────────────┐
// │  Ones              -    Three of a Kind      -   │
// │  Twos              -    Four of a Kind       -   │
// │  Threes            -    Full House           -   │
// │  Fours            12    Small Straight       -   │
// │  Fives             -    Large Straight      40   │
// │  Sixes             -    Yahtzee              -   │
// │                         Chance               -   │
// ├──────────────────────────────────────────────────┤
// │  Upper:  12 / 63    (51 to +35 bonus)            │
// │  Yahtzee bonuses:  0                             │
// │  Total:  52                                      │
// └──────────────────────────────────────────────────┘
//
// Empty boxes ("-") render dim. The upper-bonus indicator goes green once
// crossed, bold on the grand total. ANSI escapes don't count toward the
// visible width, so we pad with `pad_visible` (which knows about the styled
// vs. plain width split).

fn print_card<W: Write>(card: &Scorecard, style: &Style, output: &mut W) -> Result<()> {
    // Top border with embedded title.
    let title = " Scorecard ";
    let dashes = INNER_W.saturating_sub(title.len() + 1); // 1 for the leading "─"
    writeln!(output, "┌─{title}{}┐", "─".repeat(dashes))?;

    // 7 rows of category cells, upper-section on the left, lower-section on
    // the right. Row 7 has no left counterpart (upper has 6, lower has 7).
    for i in 0..7 {
        let left = if i < 6 {
            cell(DISPLAY_NAMES[i], card.entries[i], style)
        } else {
            (" ".repeat(CELL_W), CELL_W)
        };
        let right = cell(DISPLAY_NAMES[6 + i], card.entries[6 + i], style);
        writeln!(output, "│{}{}│", left.0, right.0)?;
    }

    // Inner divider.
    writeln!(output, "├{}┤", "─".repeat(INNER_W))?;

    // Totals.
    let upper = card.upper_total();
    let upper_status = if upper >= UPPER_BONUS_THRESHOLD {
        style.green(&format!("✓ +{UPPER_BONUS} bonus"))
    } else {
        format!(
            "({} to +{UPPER_BONUS} bonus)",
            UPPER_BONUS_THRESHOLD - upper
        )
    };
    let upper_visible = if upper >= UPPER_BONUS_THRESHOLD {
        format!("✓ +{UPPER_BONUS} bonus").chars().count()
    } else {
        format!("({} to +{UPPER_BONUS} bonus)", UPPER_BONUS_THRESHOLD - upper)
            .chars()
            .count()
    };
    let upper_line = format!("  Upper:  {upper:>2} / 63    {upper_status}");
    let upper_visible_w = "  Upper:  XX / 63    ".chars().count() + upper_visible;
    writeln!(output, "│{}│", pad_visible(&upper_line, upper_visible_w, INNER_W))?;

    let yb_line = format!(
        "  Yahtzee bonuses:  {}{}",
        card.yahtzee_bonus_count,
        if card.yahtzee_bonus_count > 0 {
            format!(" (+{})", card.yahtzee_bonus_total())
        } else {
            String::new()
        },
    );
    let yb_w = yb_line.chars().count();
    writeln!(output, "│{}│", pad_visible(&yb_line, yb_w, INNER_W))?;

    let total_value = card.grand_total();
    let total_styled = style.bold(&total_value.to_string());
    let total_line = format!("  Total:  {total_styled}");
    let total_visible_w = "  Total:  ".chars().count() + total_value.to_string().chars().count();
    writeln!(output, "│{}│", pad_visible(&total_line, total_visible_w, INNER_W))?;

    writeln!(output, "└{}┘", "─".repeat(INNER_W))?;
    Ok(())
}

/// Render a single category cell (`(text, visible_width)`). The visible width
/// is always `CELL_W` regardless of styling; we return it explicitly so the
/// caller doesn't have to strip ANSI escapes to align things.
fn cell(label: &str, value: Option<u8>, style: &Style) -> (String, usize) {
    let v_str = match value {
        Some(v) => v.to_string(),
        None => "-".to_string(),
    };
    // {:<LABEL_W$} left-pads, {:>VALUE_W$} right-aligns. Because both are
    // ASCII-only here, char count == byte count == display width.
    let plain = format!(
        "  {:<label_w$}  {:>value_w$}  ",
        label,
        v_str,
        label_w = LABEL_W,
        value_w = VALUE_W,
    );
    debug_assert_eq!(plain.chars().count(), CELL_W);
    let rendered = if value.is_none() {
        style.dim(&plain)
    } else {
        plain
    };
    (rendered, CELL_W)
}

/// Right-pad a (possibly styled) string with spaces so its visible width
/// reaches `target`. `visible` is the caller-known display width of `s`,
/// which may differ from `s.chars().count()` if `s` contains ANSI escapes.
fn pad_visible(s: &str, visible: usize, target: usize) -> String {
    if visible >= target {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(target - visible))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `DiceCounts` are looked up against a precomputed 5-of-something table,
    /// so test inputs must always sum to exactly 5. This helper makes the
    /// "three Xs and two Ys" pattern readable.
    fn dice(counts: [u8; 6]) -> DiceCounts {
        let total: u8 = counts.iter().sum();
        assert_eq!(total, 5, "dice must sum to 5, got {counts:?}");
        DiceCounts(counts)
    }

    #[test]
    fn scorecard_state_tracks_entries_and_upper() {
        let mut card = Scorecard::default();
        // Roll [1,1,1,2,3] scored in ones → 3 in the ones box.
        card.apply(0, &dice([3, 1, 1, 0, 0, 0]));
        // Yahtzee of 6s scored in sixes → 30 in the sixes box.
        card.apply(5, &dice([0, 0, 0, 0, 0, 5]));
        let s = card.state();
        assert!(s.entries.contains(ENTRY_ACTIONS[0]));
        assert!(s.entries.contains(ENTRY_ACTIONS[5]));
        assert!(!s.entries.contains(ENTRY_ACTIONS[1]));
        // upper total = 3 + 30 = 33; remaining for bonus = 63 - 33 = 30.
        assert_eq!(card.upper_total(), 33);
        assert_eq!(s.upper_score_remaining, 30);
        assert_eq!(card.upper_bonus(), 0);
    }

    #[test]
    fn upper_bonus_triggers_at_63() {
        let mut card = Scorecard::default();
        // Yahtzee of each upper face fills its box for the maximum value
        // (5×face). 5+10+15+20+25+30 = 105, well over the 63 threshold.
        card.apply(0, &dice([5, 0, 0, 0, 0, 0])); // ones:   5
        card.apply(1, &dice([0, 5, 0, 0, 0, 0])); // twos:   10
        card.apply(2, &dice([0, 0, 5, 0, 0, 0])); // threes: 15
        card.apply(3, &dice([0, 0, 0, 5, 0, 0])); // fours:  20
        card.apply(4, &dice([0, 0, 0, 0, 5, 0])); // fives:  25
        assert!(card.upper_total() >= 63);
        assert_eq!(card.upper_bonus(), 35);
    }

    #[test]
    fn yahtzee_in_yahtzee_box_sets_was_real() {
        let mut card = Scorecard::default();
        let result = card.apply(YAHTZEE_IDX, &dice([0, 0, 0, 0, 0, 5]));
        assert_eq!(result.entry_value, 50);
        assert_eq!(result.yahtzee_bonus, 0); // first yahtzee, no bonus
        assert!(card.yahtzee_was_real);
        assert_eq!(card.yahtzee_bonus_count, 0);
    }

    #[test]
    fn second_yahtzee_after_real_yahtzee_earns_bonus() {
        let mut card = Scorecard::default();
        card.apply(YAHTZEE_IDX, &dice([0, 0, 0, 0, 0, 5])); // yahtzee of 6s
        // Second yahtzee, this time of 1s, scored in ones. Joker rule N/A
        // for upper categories — gets the natural face-sum value: 5.
        let result = card.apply(0, &dice([5, 0, 0, 0, 0, 0]));
        assert_eq!(result.entry_value, 5);
        assert_eq!(result.yahtzee_bonus, 100);
        assert_eq!(card.yahtzee_bonus_count, 1);
    }

    #[test]
    fn yahtzee_zeroed_does_not_grant_bonus_eligibility() {
        let mut card = Scorecard::default();
        // Non-yahtzee dice [1,2,3,4,5] zeroed in YAHTZEE → 0, was_real stays false.
        card.apply(YAHTZEE_IDX, &dice([1, 1, 1, 1, 1, 0]));
        assert!(!card.yahtzee_was_real);
        // Later rolling a yahtzee gives no bonus.
        let result = card.apply(0, &dice([5, 0, 0, 0, 0, 0]));
        assert_eq!(result.yahtzee_bonus, 0);
        assert_eq!(card.yahtzee_bonus_count, 0);
    }

    #[test]
    fn is_dice_subset_handles_multiplicity() {
        assert!(is_dice_subset(&[1, 1], &[1, 1, 3, 4, 6]));
        assert!(!is_dice_subset(&[1, 1, 1], &[1, 1, 3, 4, 6])); // not enough 1s
        assert!(is_dice_subset(&[], &[1, 2, 3, 4, 5]));
        assert!(is_dice_subset(&[1, 2, 3, 4, 5], &[1, 2, 3, 4, 5]));
        assert!(!is_dice_subset(&[7], &[1, 2, 3, 4, 5])); // out of range
    }
}
