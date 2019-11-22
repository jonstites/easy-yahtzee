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
            Arg::with_name("score_sheet")
                .long("score_sheet")
                .short("s")
                .takes_value(true)
                .help("Yahtzee score sheet, 13 values, eg 2,0,,,,,,,,,,,"),
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

    let dice = parse_dice(matches.value_of("dice").expect("No dice provided"));
    let state = parse_score_sheet(matches.value_of("score_sheet").expect("no score sheet"));
    let result = scores.values(state);
    println!("{:?}", state);
    match matches.value_of("roll") {
        Some("1") => {
            for entry in result.first_keepers_score(dice) {
                println!("{:.*}\t{}", 5, entry.1, entry.0);
            }
        }
        Some("2") => {
            for entry in result.first_keepers_score(dice) {
                println!("{:.*}\t{}", 5, entry.1, entry.0);
            }
        }
        Some("3") => {
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

fn parse_score_sheet(entries_str: &str) -> State {
    let scores: Vec<&str> = entries_str.split(',').collect();
    let mut upper_score_remaining = 63_u8;
    let mut entries = EntryAction::default();
    let mut yahtzee_bonus_eligible = false;
    for (idx, score) in scores.iter().enumerate() {
        if score == &"" {
            continue
        }
        if idx == 11 {
            yahtzee_bonus_eligible = true;
        }
        entries |= ENTRY_ACTIONS[idx];
        if idx < 6 {
            let score = score.parse().expect("bad scoresheet");
            upper_score_remaining = upper_score_remaining.saturating_sub(score);
        }
        
    }

    State {
        upper_score_remaining,
        yahtzee_bonus_eligible,
        entries,
    }
}
