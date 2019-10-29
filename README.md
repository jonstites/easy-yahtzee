# Easy Yahtzee

[![Build Status](https://travis-ci.com/jonstites/sir-rolls-a-lot.svg?branch=master)](https://travis-ci.com/jonstites/sir-rolls-a-lot)


Topics: dynamic programming, graphs, combinatorics, probability, linear algebra, benchmarking, Rust, safe concurrency, web assembly...

## Yahtzee

Yahtzee is a dice-based game that contains elements of both luck and strategy.

[Wikipedia (CC BY-SA 3.0)](https://en.wikipedia.org/wiki/Yahtzee) has an excellent summary.

> The objective of the game is to score points by rolling five dice to make certain combinations. The dice can be rolled up to three times in a turn to try to make various scoring combinations and dice must remain in the box. A game consists of thirteen rounds. After each round the player chooses which scoring category is to be used for that round. Once a category has been used in the game, it cannot be used again. The scoring categories have varying point values, some of which are fixed values and others for which the score depends on the value of the dice. A Yahtzee is five-of-a-kind and scores 50 points, the highest of any category. The winner is the player who scores the most points. 

## Literature Search


Novel: nothing? Maybe correctness / exactness?
Performance?

buy that Yahtzee book?

## Design
Object: maximize expected score.

Goals:
correctness (what does this mean?)
performance (what kinds?)
security (...?)
ease-of-use

non-goals:
pretty?...
compilation time



### User interaction

There are many possibilities for allowing a user to interact with this project.

1. Library
2. Command-line usage only.
3. Some kind of native GUI.
4. Some kind of phone-based app.
5. Via web-browser that queries a running server.
6. Via web-browser that performs calculations on the client-side

#### Library

Pros:
- extensible, can be used however the user wants
- can be used by me, too

Cons:
- none

#### Command-line usage

Pros: 
- simple design
- very low overhead, small binaries

Cons:
- not very ergonomic


#### Native GUI

Pros:
- very flexible UI

Cons:
- very complicated
- dramatically larger binaries, most likely

#### App

Pros:
- convenience
- does this even exist?

Cons:
- complicated
- distraction from the fun/interesting bits

#### Web-browser, server-side calculations

Pros:
- extremely convenient
- can even be done without javascript
- high flexibility with UI

Cons:
- network connection required
- I'd have to have a server 24/7, or use some kind of Cloud Function
- bloated binaries?

#### Web-browser, client-side calculations

Pros:
- extremely convenient
- once code is cached, no network connection required
- faster than server-side?
- fun with WebAssembly?
- very high flexibility with UI

Cons:
- how much bandwidth required for loading initial values?
- javascript
- bloated binaries?

#### Conclusion

I'm just going to reject the phone-app and native GUI. I'm not particularly interested in learning either one at the moment, so it just seems like a lot more work than necessary. It would distract from the primary goals of this project.

At a minimum, there _should_ be a terminal-only interface. This should be compatible with a very stripped-down, small binary, and it should be fast.

Ideas - either it could be command-line argument only, or it could use ncurses, or both. Empirically, will need to see what kind of binary size is required for supporting ncurses vs command line arguments.

Probably there should _also_ be a browser version too. This makes it convenient for e.g. potential employers to verify that the thing works, to try it out, etc. And of course, there are more people who use browsers than terminals.

Client vs server. 

So I think it will come down to an empirical question - how fast can the client-side be? How fast is the server-side version? How big are the files that need to be loaded on the client side?

Also, I'd need to figure out where I would even put server-side code...I would rather not have to pay for a server, nor host something from home.

Note: regarding sending the expected values to the client. It seems like there's potential for compressing the values down. Maybe the same answers could be gotten from 8 bit values, rather than 32-bit floats or 64-bit floats. This question of lossy compression of numbers that have a very non-random distribution seems a little like image compression - maybe there is something interesting going on there.

### Library Design

So, the library should support the ability to score a state, score a state plus a given die roll, and score an action. Plus, it should be possible to choose the best action for a given state. This is just the minimum functionality required for the project.

It is also going to need to be able to load the state scores from a file. It will take minutes to hours to calculate otherwise.

Proposed design:

dependency graph

```rust
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;

use optizee::prelude::*;

fn main() -> Result<(), Box<std::error::Error>> {
    // Generate scores from scratch, may take up to hours
    let scores = Scores::new();

    let mut file = File::open("scores.ytz")?;
    let mut buf_reader = BufReader::new(file);
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer)?;

    // This could fail if the bytes are not valid representation of the scores
    let scores = Scores::from_bytes(buffer)?;


    // Returns the expected value of a yahtzee state
    let game_state = State {ones: true, fives: true, left_in_upper_score: 40, ...State::Default};
    scores.state(&game_state);

    // Returns the expected value of all the dice that could be chosen
    // TODO: decide on numeric type, look up actual size of array
    // or maybe it should return a vec of ([u8; 6], f32) ie (dice, value)
    // Returns None if an input is invalid.
    let dice = [0, 2, 0, 2, 0];
    let rolls_left = 2;
    let dice_values = scores.dice(&game_state, &dice, rolls_left).unwrap();

    // Returns the expected value of all entries that could be chosen
    let entry_values = scores.entries(&game_state, &dice).unwrap();
    Ok(())
}
```

### web browser design ?

## Math

optimizations: definitely try a matrix library. 
everything can be matrix... even the entry / child scoring part!
Multiprocessing

First we need to check that this is even a trackable problem.

So how many states are there? We will need this number to fit into memory for any reasonable algorithm.

The minimal Yahtzee state is the set of entries that have been filled, the remaining points until the upper bonus score, and whether the state is eligible for the Yahtzee bonus.

This comes to 2^13 * 63 * 2 = 1,032,192. Totally tractable. Plus - optimization! - not all of these are valid (eg how could you be Yahtzee bonus eligible before even having rolled?).

Briefly, the possibility of storing all full states - with dice and everything - is farrr too large.

So instead, we'll use these ~ 1 million intermediate states and simply calculate from there.

The expected value of a particular state is, if you play optimally, the sum of the probablility of landing in another state times its expected value.

Let's work this out on a 2-dice example. I'll show both the math-version and the code-version... hopefully...

Graph of 2-dice levels, upper score only:

Working backwards, dynamic programming...


## Internals

Scoring entries. This, it turns out, can probably be highly efficient. Matrix below!

(show scoring matrix)

Scores, besides bonuses, are determined by the dice only.

Can bonuses be matrices too?




## Biggest design uncertainties

So the biggest uncertainty around the design is the web-browser version. Should this be entirely static plus webassembly, static plus calls to backend server, or something else? 

Question: how big would a file of the expected values be? can this be shrunk down more? I would guess that best-practices put this on the sub-Megabyte range... if it ends up being larger, shouldn't I switch to something else? Such as, maybe, Cloudflare Cloud Workers?

## Tests

## Benchmarks

It will be important to benchmark:
- time to generate valid states. is this even worth it? is this impossible to multithread?
- time to perform one widget. show / calculate both the vector version, the matrix version, and ... the... all-matrix version with scoring and bonus too?
- 2^20 states vs 2^14*53 states
- time to score one entry


Binary size

compile time



## Other thoughts

I should follow this:
https://rust-lang-nursery.github.io/api-guidelines/

And I should look here for ideas (eg logging, error reporting, etc):
https://rust-lang-nursery.github.io/cli-wg/tutorial/errors.html