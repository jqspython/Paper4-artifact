import http from 'k6/http';

// Steady calibration only. This executor does not implement Poisson replay.
export const options = {
  discardResponseBodies: true,
  scenarios: {steady: {
    executor: 'constant-arrival-rate',
    rate: Number(__ENV.RATE), timeUnit: '1s', duration: '120s',
    preAllocatedVUs: 64, maxVUs: 64, gracefulStop: '5s',
  }},
};

export default function () {
  http.get(__ENV.TARGET + '/work', {timeout: '3s'});
}

export function handleSummary(data) {
  console.log('PAPER4_SUMMARY=' + JSON.stringify(data));
  return {};
}
