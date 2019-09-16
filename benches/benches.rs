#![feature(test)]

extern crate optizee;

mod bench {
    extern crate test;
    use self::test::Bencher;

    #[bench]
    fn bench_valid_states(b: &mut Bencher) {
        b.iter(|| {
           test::black_box(optizee::valid_states());        
        });
    }
}