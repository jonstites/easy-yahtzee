
use std::convert::From;
use std::fmt;

#[macro_use]
extern crate itertools;
use itertools::Itertools;

#[macro_use]
extern crate lazy_static;

const NUM_STATES: usize = 1048576;



lazy_static! {
    static ref UPPER_SCORES: [[usize; 252]; 13] = {
        let mut uppers = [[0; 252]; 13];
        let mut idx = 0;
        'outer:
        for dice in iproduct!(0..6, 0..6, 0..6, 0..6, 0..6) {
            let dice = vec!(dice.0, dice.1, dice.2, dice.3, dice.4);
            for die in 1..5 {
                if dice[die] < dice[die - 1] {
                    continue 'outer;
                }
            }
            for action in 0..6 {
                let num_die = dice.iter().filter(|die| **die == action).count();
                
                let score = num_die * (action + 1);
                println!("{:?} {:?} {:?} {:?}", dice, num_die, action, score);
                uppers[action][idx] = score;
            }
            idx += 1;
        }
        uppers
    };

    static ref YAHTZEE: [bool; 252] = {
        let mut yahtzees = [false; 252];
        let mut idx = 0;
        'outer:
        for dice in iproduct!(0..6, 0..6, 0..6, 0..6, 0..6) {
            let dice = vec!(dice.0, dice.1, dice.2, dice.3, dice.4);
            for die in 1..5 {
                if dice[die] < dice[die - 1] {
                    continue 'outer;
                }
            }
            idx += 1;

            for j in 1..5 {
                if dice[j] != dice[j - 1] {
                    continue 'outer;
                }
            }

            yahtzees[idx - 1] = true;
        }
        yahtzees
    };
}


#[derive(Debug, Copy, Clone)]
struct State(usize);

impl Default for State {

    fn default() -> State {
        State(53_usize)
    }
}

impl fmt::Display for State {

    fn fmt(&self, dest: &mut fmt::Formatter) -> fmt::Result {
        write!(dest, "{:#b}", self.0)
    }

}

impl State {
    fn children(&self) -> Box<[[Option<State>; 252]]> {
        let mut children_values = vec![[None; 252]; 13];

        for i in 0..13 {
            // if this action hasn't been taken yet            
            if ((self.0 >> (i + 7)) & 1) != 1 {

                for j in 0..252 {                    
                    let mut child = *self;

                    // set action
                    child.0 |= 1 << (i + 7);

                    // set upper score
                    let upper_score = (child.0 & 0b11_1111).saturating_sub(UPPER_SCORES[i][j]);
                    child.0 = (child.0 >> 6) << 6;
                    child.0 |= upper_score;
                    
                    // set yahtzee eligibility
                    if i == 11 && YAHTZEE[j] {
                        child.0 |= 1 << 7;
                    }
                    children_values[i][j] = Some(child);
                }
            }
        }
        children_values.into_boxed_slice()
    }
}

impl From<usize> for State {
    fn from(value: usize) -> Self {
        let value = value.min(NUM_STATES - 1);
        State(value)
    }
}

impl From<State> for usize {
    fn from(value: State) -> Self {
        value.0
    }
}

pub fn valid_states() -> Box<[bool]> {

    let mut valid_markers = vec![false;  NUM_STATES];

    valid_markers[0] = true;
    let mut queue = vec!(State(0b11111 << 7));
    for i in 0..UPPER_SCORES.len() {
        for j in 0..UPPER_SCORES[i].len() {
            println!("{:?} {:?} {:?}", i, j, UPPER_SCORES[i][j]);
        }
    }
    while let Some(elem) = queue.pop() {        

        for child in elem.children().iter().flat_map(|array| array.iter()).filter_map(|child| *child) {

            let idx: usize = child.into();
            if !valid_markers[idx] {
                //println!("{} {}", elem, child);
                valid_markers[idx] = true;
                queue.push(child);
            }
        }
    }
    valid_markers.into_boxed_slice()
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_valid_states() {
        let states = valid_states();
        let num_valid = states.iter().filter(|x| **x).count();
        assert_eq!(num_valid, 536448);
    }
}