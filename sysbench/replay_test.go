package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestPlanAndFailedLatency(t *testing.T) {
	path := filepath.Join(t.TempDir(), "plan.csv")
	os.WriteFile(path, []byte("request_id,offset_ns\n0,0\n1,100\n"), 0600)
	r, h, e := readPlan(path, 1000)
	if e != nil || len(r) != 2 || len(h) != 64 || r[1].Written != -1 {
		t.Fatal(r, h, e)
	}
	os.WriteFile(path, []byte("request_id,offset_ns\n1,0\n"), 0600)
	if _, _, e = readPlan(path, 1000); e == nil {
		t.Fatal("accepted missing request id")
	}
	if !math.IsInf(quantile([]float64{1, 2, 3, math.Inf(1)}, .95), 1) {
		t.Fatal("lost failed request from p95")
	}
	if monoNS() <= 0 || cpuSeconds() < 0 {
		t.Fatal("invalid monotonic or process CPU clock")
	}
}
