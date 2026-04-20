use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use yahtzee_core::{DiceCounts, EntryAction, Scores, State, ENTRY_ACTIONS};

#[wasm_bindgen]
pub struct Solver {
    inner: Scores,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct StateInput {
    pub entries: [bool; 13],
    pub yahtzee_bonus_eligible: bool,
    pub upper_score_remaining: u8,
}

#[derive(Serialize, Clone)]
pub struct KeeperRec {
    pub dice: Vec<u8>,
    pub ev: f32,
}

#[derive(Serialize, Clone)]
pub struct EntryRec {
    pub entry: &'static str,
    pub ev: f32,
}

#[derive(Serialize, Clone)]
pub struct Recommendation {
    pub value: f32,
    pub keepers: Option<Vec<KeeperRec>>,
    pub entries: Option<Vec<EntryRec>>,
}

#[wasm_bindgen]
impl Solver {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<Solver, JsValue> {
        let inner: Scores = bincode::deserialize(bytes)
            .map_err(|e| JsValue::from_str(&format!("deserialize failed: {}", e)))?;
        Ok(Solver { inner })
    }

    pub fn recommend(
        &self,
        state_js: JsValue,
        dice: Vec<u8>,
        roll: u8,
    ) -> Result<JsValue, JsValue> {
        let input: StateInput = serde_wasm_bindgen::from_value(state_js)?;
        let rec = core_recommend(&self.inner, &input, &dice, roll).map_err(js_err)?;
        serde_wasm_bindgen::to_value(&rec).map_err(Into::into)
    }

    #[wasm_bindgen(js_name = stateValue)]
    pub fn state_value(&self, state_js: JsValue) -> Result<f32, JsValue> {
        let input: StateInput = serde_wasm_bindgen::from_value(state_js)?;
        let state = build_state(&input).map_err(js_err)?;
        Ok(self.inner.values(state).value)
    }
}

pub fn core_recommend(
    scores: &Scores,
    input: &StateInput,
    dice: &[u8],
    roll: u8,
) -> Result<Recommendation, String> {
    let state = build_state(input)?;
    let dice_counts = dice_to_counts(dice)?;
    let values = scores.values(state);

    Ok(match roll {
        1 => Recommendation {
            value: values.value,
            keepers: Some(
                values
                    .first_keepers_score(dice_counts)
                    .into_iter()
                    .map(|(d, ev)| KeeperRec {
                        dice: counts_to_faces(&d),
                        ev,
                    })
                    .collect(),
            ),
            entries: None,
        },
        2 => Recommendation {
            value: values.value,
            keepers: Some(
                values
                    .second_keepers_score(dice_counts)
                    .into_iter()
                    .map(|(d, ev)| KeeperRec {
                        dice: counts_to_faces(&d),
                        ev,
                    })
                    .collect(),
            ),
            entries: None,
        },
        3 => Recommendation {
            value: values.value,
            keepers: None,
            entries: Some(
                values
                    .entries_score(dice_counts)
                    .into_iter()
                    .map(|(a, ev)| EntryRec {
                        entry: action_name(a),
                        ev,
                    })
                    .collect(),
            ),
        },
        _ => return Err("roll must be 1, 2, or 3".into()),
    })
}

fn build_state(input: &StateInput) -> Result<State, String> {
    let mut entries = EntryAction::empty();
    for (i, &on) in input.entries.iter().enumerate() {
        if on {
            entries |= ENTRY_ACTIONS[i];
        }
    }
    if input.upper_score_remaining > 63 {
        return Err("upper_score_remaining must be 0..=63".into());
    }
    Ok(State {
        entries,
        yahtzee_bonus_eligible: input.yahtzee_bonus_eligible,
        upper_score_remaining: input.upper_score_remaining,
    })
}

fn dice_to_counts(dice: &[u8]) -> Result<DiceCounts, String> {
    if dice.len() != 5 {
        return Err(format!("dice must have length 5, got {}", dice.len()));
    }
    let mut counts = [0_u8; 6];
    for &d in dice {
        if !(1..=6).contains(&d) {
            return Err(format!("each die must be 1..=6, got {}", d));
        }
        counts[(d - 1) as usize] += 1;
    }
    Ok(DiceCounts(counts))
}

fn counts_to_faces(dc: &DiceCounts) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, &c) in dc.0.iter().enumerate() {
        for _ in 0..c {
            out.push((i + 1) as u8);
        }
    }
    out
}

fn action_name(a: EntryAction) -> &'static str {
    match a {
        EntryAction::ONE => "ones",
        EntryAction::TWO => "twos",
        EntryAction::THREE => "threes",
        EntryAction::FOUR => "fours",
        EntryAction::FIVE => "fives",
        EntryAction::SIX => "sixes",
        EntryAction::THREE_OF_A_KIND => "three_of_a_kind",
        EntryAction::FOUR_OF_A_KIND => "four_of_a_kind",
        EntryAction::FULL_HOUSE => "full_house",
        EntryAction::SMALL_STRAIGHT => "small_straight",
        EntryAction::LARGE_STRAIGHT => "large_straight",
        EntryAction::YAHTZEE => "yahtzee",
        EntryAction::CHANCE => "chance",
        _ => "unknown",
    }
}

fn js_err(s: String) -> JsValue {
    JsValue::from_str(&s)
}
