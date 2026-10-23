// D10 only: constant-rate, planned-to-completion latency, same origin as Go.
// Formal workloads remain immutable Poisson schedules in the Go replayer.
import http from 'k6/http';
import exec from 'k6/execution';
import { Trend, Counter } from 'k6/metrics';

const rate = 96;
const plannedLatency = new Trend('planned_latency_ms', true);
const failed = new Counter('crosscheck_failures');
export const options = {
  scenarios: { cross: { executor: 'constant-arrival-rate', rate,
    timeUnit: '1s', duration: '120s', preAllocatedVUs: 64, maxVUs: 64,
    gracefulStop: '1s' } },
  discardResponseBodies: true,
};

export default function () {
  const plannedMs = exec.scenario.iterationInTest * 1000 / rate;
  // The arrival executor uses its scenario origin, which can differ from the
  // global test origin. progress is computed from that same monotonic origin.
  const scenarioStartMs = exec.scenario.progress * 120000;
  const before = exec.instance.currentTestRunDuration;
  const response = http.get(__ENV.TARGET + '/work', { timeout: '189.2ms' });
  const elapsed = scenarioStartMs - plannedMs + exec.instance.currentTestRunDuration - before;
  if (response.status < 200 || response.status >= 300 || response.error ||
      !Number.isFinite(elapsed) || elapsed < 0 || elapsed > 189.2) {
    failed.add(1);
  }
  // A crosscheck with any error/drop is ineligible; formal D3 uses +Inf.
  plannedLatency.add(elapsed);
}

export function handleSummary(data) {
  return { stdout: 'PAPER4_SUMMARY=' + JSON.stringify(data) + '\n' };
}
