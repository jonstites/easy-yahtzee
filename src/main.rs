extern crate optizee;

fn main() {
    let scores = optizee::Scores::new();
    println!("{:#?}", scores.values(optizee::State::default()));
}
