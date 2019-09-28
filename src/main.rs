extern crate optizee;

fn main() {
    println!("starting!");
    let idx: usize = optizee::State::default().into();
    println!("{:?}", optizee::dice_probability(&[1, 1, 0, 0, 0, 0]));
    println!("{:?}", optizee::dice_probability(&[2, 0, 0, 0, 0, 0]));
    println!("{:?}", optizee::scores()[idx]);
    /*for val in optizee::dice_combinations2(5) {
        println!("{:?}", val);
    }*/
}
