use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use yahtzee_core::{
    achievable_scores, DiceCounts, EntryAction, Scores, State, ENTRY_ACTIONS,
};

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
    pub turn_ev: f32,
}

#[derive(Serialize, Clone)]
pub struct EntryRec {
    /// Index into `ENTRY_ACTIONS` (0 = ones, …, 12 = chance). The UI maps this
    /// directly to its label/lookup arrays — no need to round-trip a string.
    pub entry: u8,
    pub ev: f32,
    pub turn_ev: f32,
}

#[derive(Serialize, Clone)]
pub struct Recommendation {
    pub value: f32,
    /// Per-keeper rankings on rolls 1 and 2; `None` on roll 3 (no more rolls).
    /// The all-5 keeper is intentionally excluded: "keep all 5 then re-roll
    /// some later" is strictly dominated by "re-roll those now". The only
    /// useful "keep everything" interpretation is "score these dice now",
    /// which the `entries` field captures with the actual score-now EV.
    pub keepers: Option<Vec<KeeperRec>>,
    /// Per-entry rankings, populated on every roll. On roll 3 these are the
    /// only options. On rolls 1 and 2 they represent "skip the remaining
    /// rolls and score this dice in this category", with `ev` / `turn_ev`
    /// reflecting that decision (no further rolls), so the UI can merge them
    /// with the keeper list and sort by overall EV.
    pub entries: Vec<EntryRec>,
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

    #[wasm_bindgen(js_name = thisTurnEv)]
    pub fn this_turn_ev(
        &self,
        state_js: JsValue,
        dice: Vec<u8>,
        roll: u8,
    ) -> Result<f32, JsValue> {
        let input: StateInput = serde_wasm_bindgen::from_value(state_js)?;
        let state = build_state(&input).map_err(js_err)?;
        let dice_counts = dice_to_counts(&dice).map_err(js_err)?;
        if !(1..=3).contains(&roll) {
            return Err(JsValue::from_str("roll must be 1, 2, or 3"));
        }
        Ok(self.inner.values(state).this_turn_ev(dice_counts, roll))
    }
}

#[wasm_bindgen(js_name = entryScore)]
pub fn entry_score_js(state_js: JsValue, dice: Vec<u8>, entry_idx: u8) -> Result<u8, JsValue> {
    let input: StateInput = serde_wasm_bindgen::from_value(state_js)?;
    let state = build_state(&input).map_err(js_err)?;
    let dice_counts = dice_to_counts(&dice).map_err(js_err)?;
    let idx = entry_idx as usize;
    if idx >= ENTRY_ACTIONS.len() {
        return Err(JsValue::from_str("entry_idx out of range"));
    }
    Ok(state.entry_score(ENTRY_ACTIONS[idx], &dice_counts))
}

#[wasm_bindgen(js_name = achievableScores)]
pub fn achievable_scores_js(entry_idx: u8) -> Result<Vec<u8>, JsValue> {
    let idx = entry_idx as usize;
    if idx >= ENTRY_ACTIONS.len() {
        return Err(JsValue::from_str("entry_idx out of range"));
    }
    Ok(achievable_scores(idx))
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

    let entries: Vec<EntryRec> = values
        .entries_with_turn_ev(dice_counts.clone())
        .into_iter()
        .map(|(a, ev, turn_ev)| EntryRec {
            entry: action_idx(a),
            ev,
            turn_ev,
        })
        .collect();

    let keepers = match roll {
        1 => Some(
            values
                .first_keepers_with_turn_ev(dice_counts)
                .into_iter()
                .filter(|(d, _, _)| keeper_size(d) < 5)
                .map(|(d, ev, turn_ev)| KeeperRec {
                    dice: counts_to_faces(&d),
                    ev,
                    turn_ev,
                })
                .collect(),
        ),
        2 => Some(
            values
                .second_keepers_with_turn_ev(dice_counts)
                .into_iter()
                .filter(|(d, _, _)| keeper_size(d) < 5)
                .map(|(d, ev, turn_ev)| KeeperRec {
                    dice: counts_to_faces(&d),
                    ev,
                    turn_ev,
                })
                .collect(),
        ),
        3 => None,
        _ => return Err("roll must be 1, 2, or 3".into()),
    };

    Ok(Recommendation {
        value: values.value,
        keepers,
        entries,
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

fn keeper_size(dice: &DiceCounts) -> u16 {
    dice.0.iter().map(|&c| c as u16).sum()
}

/// Index of an `EntryAction` in `ENTRY_ACTIONS` (0..13). Panics on a
/// non-canonical action; in practice the core only ever hands us actions from
/// `ENTRY_ACTIONS` so this is unreachable.
fn action_idx(a: EntryAction) -> u8 {
    ENTRY_ACTIONS
        .iter()
        .position(|&x| x == a)
        .expect("unknown EntryAction") as u8
}

fn js_err(s: String) -> JsValue {
    JsValue::from_str(&s)
}
