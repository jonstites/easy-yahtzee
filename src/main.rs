extern crate optizee;

fn main() {
    println!("starting!");
    let idx: usize = optizee::State::default().into();
    println!("{:?}", optizee::scores()[idx]);
}
