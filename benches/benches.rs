#![feature(test)]

extern crate optizee;

mod bench {
    extern crate test;
    use self::test::Bencher;

    #[bench]
    fn bench_valid_states(b: &mut Bencher) {
        b.iter(|| {
            let n = test::black_box(100);

            for action_idx in 0..13 {
                for dice_idx in 0..252 {
                    for state in 0..n {
                        let state: optizee::State = state.into();
                        state.child(action_idx, dice_idx);
                    }
                }
            }
        });
    }

    #[bench]
    fn bench_widget(b: &mut Bencher) {
        let scores = vec![1_f64; 1048576];
        let f = test::black_box(optizee::widget(optizee::State::default(), &scores));        
        b.iter(|| {
            let _f = f;
            let n = test::black_box(100);
            for state in 0..n {
                let state: optizee::State = state.into();
                optizee::widget(state, &scores);
            }
        });
    }

}