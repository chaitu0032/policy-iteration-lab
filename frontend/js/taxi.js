// Taxi state encoding (mirrors gymnasium's TaxiEnv.encode / decode).

export const TAXI_IN_CAR = 4;
export const PASSENGER_LABELS = ["R", "G", "Y", "B", "In taxi"];

export function encodeTaxi(row, col, passenger, destination) {
  return ((row * 5 + col) * 5 + passenger) * 4 + destination;
}

export function decodeTaxi(state) {
  const destination = state % 4;
  let rest = Math.floor(state / 4);
  const passenger = rest % 5;
  rest = Math.floor(rest / 5);
  return { row: Math.floor(rest / 5), col: rest % 5, passenger, destination };
}

export function describeState(envKey, layout, state) {
  if (envKey === "Taxi") {
    const d = decodeTaxi(state);
    return `s${state} · taxi (${d.row},${d.col}) · passenger ${PASSENGER_LABELS[d.passenger]} → ${layout.loc_names[d.destination]}`;
  }
  const r = Math.floor(state / layout.cols);
  const c = state % layout.cols;
  return `s${state} · (${r},${c})`;
}
