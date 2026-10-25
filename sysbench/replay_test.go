package main

import (
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestHealthSetupDeadlineAndHTTPFailureStayFailures(t *testing.T) {
	for _, mode := range []string{"deadline", "http"} {
		t.Run(mode, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if mode == "deadline" {
					<-r.Context().Done()
					return
				}
				w.WriteHeader(503)
			}))
			defer server.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()
			probes := 0
			err := warmHealth(ctx, server.Client(), server.URL, 25, func(v any) {
				e := v.(map[string]any)
				probes++
				if e["type"] != "bootstrap_probe" || e["outcome"] != "failure" {
					t.Fatal(e)
				}
			})
			if err == nil || probes != 1 {
				t.Fatal("unbounded or swallowed setup failure", err, probes)
			}
		})
	}
}

// Inject exactly the observed symptom, not a claim about the old network cause.
func TestT1TransientHealthTimeoutMustNotDiscardAllPreconnections(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 26 {
			<-r.Context().Done()
			return
		}
		w.Header().Set("X-Paper4-Pod", "offline-pod")
		w.Write([]byte("ready\n"))
	}))
	defer server.Close()
	ready := false
	clients, closeAll, err := independentClients(server.URL, 256, func(v any) {
		if e := v.(map[string]any); e["type"] == "transport_ready" {
			ready = true
		}
	})
	defer closeAll()
	if err != nil || len(clients) != 256 || !ready || calls.Load() != 257 {
		t.Fatalf("transient pre-clock timeout discarded T1: clients=%d ready=%v calls=%d error=%v", len(clients), ready, calls.Load(), err)
	}
}

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

func TestIndependentTransportsPreconnectReuseAndRedial(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Connection", r.RemoteAddr)
		w.Write([]byte("ok"))
	}))
	defer server.Close()
	var lock sync.Mutex
	var events []map[string]any
	clients, closeAll, err := independentClients(server.URL+"/work", 256, func(v any) {
		lock.Lock()
		defer lock.Unlock()
		events = append(events, v.(map[string]any))
	})
	if err != nil {
		t.Fatal(err)
	}
	defer closeAll()
	addresses := map[string]bool{}
	perSlot := make([]string, 256)
	for i := 0; i < 512; i++ {
		client := clients[i%256]
		tr := client.Transport.(*http.Transport)
		if tr.MaxConnsPerHost != 1 || tr.MaxIdleConnsPerHost != 1 {
			t.Fatal("not a single-connection transport")
		}
		resp, e := client.Get(server.URL + "/work")
		if e != nil {
			t.Fatal(e)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		address := resp.Header.Get("X-Connection")
		if i < 256 {
			perSlot[i] = address
			addresses[address] = true
		} else if perSlot[i%256] != address {
			t.Fatal("connection not reused")
		}
	}
	if len(addresses) != 256 {
		t.Fatal("transports shared connections", len(addresses))
	}
	clients[3].CloseIdleConnections()
	resp, e := clients[3].Get(server.URL + "/work")
	if e != nil {
		t.Fatal(e)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	lock.Lock()
	defer lock.Unlock()
	dials, redials, ready := 0, 0, false
	for _, event := range events {
		if event["type"] == "connection" {
			dials++
			if event["redial"] == true {
				redials++
			}
		}
		if event["type"] == "transport_ready" {
			ready = event["preconnected"] == 256
		}
	}
	if dials != 257 || redials != 1 || !ready {
		t.Fatal("missing preconnect/redial evidence", dials, redials, ready)
	}
}

func TestReadyRefreshRetainsSlotClientsAndForcesNextDial(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) }))
	defer server.Close()
	dials := 0
	clients, closeAll, err := independentClients(server.URL, 256, func(v any) {
		if v.(map[string]any)["type"] == "connection" {
			dials++
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer closeAll()
	before := clients[0]
	receipts := 0
	input := fmt.Sprintf("{\"tick\":1,\"ready_pods\":[\"a\",\"b\"],\"requested_mono_ns\":%d}\n", monoNS())
	if err = refreshIdle(strings.NewReader(input), closeAll, func(any) { receipts++ }, 60); err != nil {
		t.Fatal(err)
	}
	for i := range clients {
		response, e := clients[i].Get(server.URL)
		if e != nil {
			t.Fatal(e)
		}
		io.Copy(io.Discard, response.Body)
		response.Body.Close()
	}
	if before != clients[0] || receipts != 1 || dials != 512 {
		t.Fatal("changed clients or missing redials", receipts, dials)
	}
	if err = refreshIdle(strings.NewReader(input+input), func() {}, func(any) {}, 60); err == nil {
		t.Fatal("accepted duplicate tick")
	}
}
