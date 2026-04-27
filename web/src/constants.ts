/// Canonical scorecard order. Index here is the `entry` index used in
/// `EntryRec.entry`, `entryScore(_, _, idx)`, and `achievableScores(idx)`.
export const ENTRY_LABELS = [
  'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
  'Three of a kind', 'Four of a kind', 'Full house',
  'Small straight', 'Large straight', 'Yahtzee', 'Chance',
] as const;

export const FULL_HOUSE_IDX = 8;
export const SMALL_STRAIGHT_IDX = 9;
export const LARGE_STRAIGHT_IDX = 10;
export const YAHTZEE_IDX = 11;

/// Lower-row entries that accept the Joker rule's fixed scores (25 / 30 / 40)
/// when a Yahtzee is rolled and the matching upper box + Yahtzee box are both
/// already filled. Order matches scorecard order.
export const JOKER_LOWER_IDXS = [
  FULL_HOUSE_IDX,
  SMALL_STRAIGHT_IDX,
  LARGE_STRAIGHT_IDX,
] as const;

export const HOW_TO_SCORE = [
  'count 1s', 'count 2s', 'count 3s', 'count 4s', 'count 5s', 'count 6s',
  'sum of dice', 'sum of dice', '25', '30', '40', '50', 'sum of dice',
] as const;

export const PIP_POS: Record<number, [number, number][]> = {
  1: [[50, 50]],
  2: [[25, 25], [75, 75]],
  3: [[25, 25], [50, 50], [75, 75]],
  4: [[25, 25], [75, 25], [25, 75], [75, 75]],
  5: [[25, 25], [75, 25], [50, 50], [25, 75], [75, 75]],
  6: [[25, 25], [75, 25], [25, 50], [75, 50], [25, 75], [75, 75]],
};
