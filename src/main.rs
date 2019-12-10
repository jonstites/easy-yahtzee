use easy_yahtzee::{DiceCounts, EntryAction, Scores, State, ENTRY_ACTIONS};

use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::Write;
use std::path::Path;

#[macro_use]
extern crate clap;
use clap::{App, Arg, SubCommand};
use exitfailure::ExitFailure;
use failure::ResultExt;

fn main() -> Result<(), ExitFailure> {
    
    let matches = App::new("easy-yahtzee")
        .version(crate_version!())
	.subcommand(SubCommand::with_name("save")
		    .about("save scores to a file")
		    .arg(
			Arg::with_name("output")
			    .short("o")
			    .long("output")
			    .value_name("output")
			    .help("Regenerate scores and save to file")
			    .required(true)
			    .takes_value(true)))
	.subcommand(SubCommand::with_name("score")
		    .about("score different options")
		    .arg(
			Arg::with_name("input")
			    .short("i")
			    .long("input")
			    .value_name("input")
			    .help("Load scores from a file")
			    .required(true)
			    .takes_value(true)))
        .get_matches();

    if let Some(save_matches) = matches.subcommand_matches("save") {
	let scores = Scores::new();
	let output = save_matches.value_of("output").unwrap();
        let dest_path = Path::new(&output);
        let mut f = File::create(&dest_path)?;
        let encoded: Vec<u8> = bincode::serialize(&scores)?;
        f.write_all(&encoded[..])?;
    } else if let Some(score_matches) = matches.subcommand_matches("score") {
	let input = score_matches.value_of("input").unwrap();
        let scores_path = Path::new(&input);
        let mut scores_file = File::open(scores_path)?;
        let mut buffer = Vec::new();
        scores_file.read_to_end(&mut buffer)?;
        let scores: Scores = bincode::deserialize(&buffer)?;

	// loop through all entries, prompt for roll #
    }

    
    Ok(())
    /*
    let scores = if let Some(input) = matches.value_of("input") {
        scores
    } else {
        Scores::new()
    };

    if let Some(output) = matches.value_of("output") {
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
    entry*/
}
