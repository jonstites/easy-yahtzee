use easy_yahtzee::{DiceCounts, EntryAction, Scores, State, ENTRY_ACTIONS};

use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::Write;
use std::path::Path;

#[macro_use]
extern crate clap;
use clap::{App, Arg};

fn main() -> std::io::Result<()> {
    let matches = App::new("easy-yahtzee")
        .version(crate_version!())
        .arg(
            Arg::with_name("output")
                .short("o")
                .long("output")
                .value_name("output")
                .help("Regenerate scores and save to file")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("input")
                .short("i")
                .long("input")
                .value_name("input")
                .help("Load scores from a file")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("roll")
                .short("r")
                .long("roll")
                .possible_values(&["1", "2", "3"])
                .help("Which yahtzee roll are you on?")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("dice")
                .short("d")
                .long("dice")
                .help("What dice did you roll?")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("upper_score_remaining")
                .short("u")
                .long("upper_score_remaining")
                .help("Upper score remaining before bonus")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("yahtzee_eligible")
                .long("yahtzee_eligible")
                .help("Eligible for yahtzee bonus"),
        )
        .arg(
            Arg::with_name("entries")
                .long("entries")
                .short("e")
                .takes_value(true)
                .help("Impossible to decipher entry bitstring"),
        )
        .get_matches();

    let scores = if let Some(input) = matches.value_of("input") {
        let scores_path = Path::new(&input);
        let mut scores_file = File::open(scores_path)?;
        let mut buffer = Vec::new();
        scores_file.read_to_end(&mut buffer)?;
        let scores: Scores = bincode::deserialize(&buffer).unwrap();
        scores
    } else {
        Scores::new()
    };

    if let Some(output) = matches.value_of("output") {
        let dest_path = Path::new(&output);
        let mut f = File::create(&dest_path).unwrap();
        let encoded: Vec<u8> = bincode::serialize(&scores).unwrap();
        f.write_all(&encoded[..]).unwrap();
    }

    match matches.value_of("roll") {
        Some("1") => {
            let dice = parse_dice(matches.value_of("dice").expect("No dice provided"));
            let upper_score_remaining: u8 = matches
                .value_of("upper_score_remaining")
                .expect("no upper remaining specified")
                .parse()
                .expect("could not parse upper as u8");
            let yahtzee_bonus_eligible: bool = matches.is_present("yahtzee_bonus_eligible");
            let entries = parse_entries(matches.value_of("entries").expect("expected entries"));
            let state = State {
                upper_score_remaining,
                yahtzee_bonus_eligible,
                entries,
            };
            let result = scores.values(state);
            for entry in result.first_keepers_score(dice) {
                println!("{:.*}\t{}", 5, entry.1, entry.0);
            }
        }
        Some("2") => {
            let dice = parse_dice(matches.value_of("dice").expect("No dice provided"));
            let upper_score_remaining: u8 = matches
                .value_of("upper_score_remaining")
                .expect("no upper remaining specified")
                .parse()
                .expect("could not parse upper as u8");
            let yahtzee_bonus_eligible: bool = matches.is_present("yahtzee_bonus_eligible");
            let entries = parse_entries(matches.value_of("entries").expect("expected entries"));
            let state = State {
                upper_score_remaining,
                yahtzee_bonus_eligible,
                entries,
            };
            let result = scores.values(state);
            for entry in result.second_keepers_score(dice) {
                println!("{:.*}\t{}", 5, entry.1, entry.0);
            }
        }
        Some("3") => {
            let dice = parse_dice(matches.value_of("dice").expect("No dice provided"));
            let upper_score_remaining: u8 = matches
                .value_of("upper_score_remaining")
                .expect("no upper remaining specified")
                .parse()
                .expect("could not parse upper as u8");
            let yahtzee_bonus_eligible: bool = matches.is_present("yahtzee_bonus_eligible");
            let entries = parse_entries(matches.value_of("entries").expect("expected entries"));
            let state = State {
                upper_score_remaining,
                yahtzee_bonus_eligible,
                entries,
            };
            let result = scores.values(state);
            for entry in result.entries_score(dice) {
                println!("{:08}\t{:?}", entry.1, entry.0);
            }
        }
        _ => (),
    };

    Ok(())
}

fn parse_dice(dice_str: &str) -> DiceCounts {
    let dice: Vec<usize> = dice_str
        .split(",")
        .map(|d| d.parse().expect("bad dice provided"))
        .collect();

    if dice.len() != 5 {
        panic!("Expected 5 dice, not {}", dice.len());
    }
    let mut counts = [0; 6];
    for die in dice {
        if die > 6 || die == 0 {
            panic!("Dice out of bounds: {}", die);
        }

        counts[die - 1] += 1;
    }
    DiceCounts(counts)
}

fn parse_entries(entries_str: &str) -> EntryAction {
    let chars: Vec<char> = entries_str.chars().collect();
    if chars.len() != 13 {
        panic!("Expected entries length 13 not {}", chars.len());
    }

    let mut entry = EntryAction::default();
    for idx in 0..13 {
        match chars[idx] {
            '0' => (),
            '1' => entry |= ENTRY_ACTIONS[idx],
            _ => panic!("Expected bits, not {}", chars[idx]),
        }
    }
    entry
}
