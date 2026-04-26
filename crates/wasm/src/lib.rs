use wasm_bindgen::prelude::*;
use yahtzee_core::{
    achievable_scores, build_state, dice_to_counts, recommend, Scores, ENTRY_ACTIONS,
};

// Re-export the recommendation types so downstream Rust consumers (e.g. the
// integration tests in `tests/recommend.rs`) can name them through this crate.
pub use yahtzee_core::{EntryRec, KeeperRec, Recommendation, StateInput};

/// Re-export under the previous name so existing callers (the integration
/// test) keep working without changing import paths.
pub fn core_recommend(
    scores: &Scores,
    input: &StateInput,
    dice: &[u8],
    roll: u8,
) -> Result<Recommendation, String> {
    recommend(scores, input, dice, roll)
}

#[wasm_bindgen]
pub struct Solver {
    inner: Scores,
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
        let rec = recommend(&self.inner, &input, &dice, roll).map_err(js_err)?;
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

fn js_err(s: String) -> JsValue {
    JsValue::from_str(&s)
}
