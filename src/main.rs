extern crate optizee;

use std::collections::HashMap;

use optizee::State;
use optizee::DiceCounts;
use optizee::DiceCombinations;
use optizee::ExpectedValues;
use optizee::widget;

fn main() {

    //entries[1] = false;
    //entries[12] = false;
//    let mut entries =             [true;14];
//    entries[8] = false;
    //    entries[13] = false;
    let entries = [false; 14];

    let state = State::new(
        entries,
        0,
        3,
        DiceCounts::new([0;6]));
    
    let mut cache = ExpectedValues { ev: HashMap::new() };
    let dc = DiceCombinations::new();
        
    
    let ev = widget(state, &mut cache, &dc);
    //println!("{:?}: {:?}", i, ev);
    println!("{:?}", ev);
}
