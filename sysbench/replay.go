// Linux open-loop HTTP replay. The immutable CSV, never response completion,
// determines arrivals. WroteRequest is a conservative completed-write timestamp.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptrace"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

type record struct {
	Offset, Written, Done int64
	WriteCallbacks        int64
	Code                  int
	Error                 string
}

// The calibrated Python HTTP/1.1 server writes headers/body separately.
// Prompt client ACKs prevent a delayed-ACK/Nagle interaction from becoming a
// sender-specific 40ms measurement artifact. This does not change server work.
type ackConn struct{ *net.TCPConn }

func (c *ackConn) Read(p []byte) (int, error) {
	n, err := c.TCPConn.Read(p)
	if n > 0 {
		raw, e := c.SyscallConn()
		if e == nil {
			e = raw.Control(func(fd uintptr) {
				if inner := syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, syscall.TCP_QUICKACK, 1); inner != nil {
					err = inner
				}
			})
		}
		if e != nil {
			err = e
		}
	}
	return n, err
}

func monoNS() int64 {
	var ts syscall.Timespec
	_, _, err := syscall.RawSyscall(syscall.SYS_CLOCK_GETTIME, 1, uintptr(unsafe.Pointer(&ts)), 0)
	if err != 0 {
		panic(err)
	}
	return ts.Sec*1e9 + ts.Nsec
}

func cpuSeconds() float64 {
	var u syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &u); err != nil {
		panic(err)
	}
	return float64(u.Utime.Sec+u.Stime.Sec) + float64(u.Utime.Usec+u.Stime.Usec)/1e6
}

func readPlan(path string, duration int64) ([]record, string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, "", err
	}
	hash := sha256.Sum256(raw)
	file, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}
	defer file.Close()
	r := csv.NewReader(file)
	header, err := r.Read()
	if err != nil || len(header) != 2 || header[0] != "request_id" || header[1] != "offset_ns" {
		return nil, "", errors.New("expected request_id,offset_ns CSV")
	}
	var records []record
	previous := int64(-1)
	for {
		row, e := r.Read()
		if e == io.EOF {
			break
		}
		if e != nil {
			return nil, "", e
		}
		id, e := strconv.Atoi(row[0])
		if e != nil || id != len(records) {
			return nil, "", errors.New("nonconsecutive request identity")
		}
		ns, e := strconv.ParseInt(row[1], 10, 64)
		if e != nil || ns < 0 || ns >= duration || ns < previous {
			return nil, "", errors.New("invalid plan deadline")
		}
		records = append(records, record{Offset: ns, Written: -1, Done: -1})
		previous = ns
	}
	return records, hex.EncodeToString(hash[:]), nil
}

// Nearest-rank quantiles; +Inf failures remain in latency denominators.
func quantile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return math.NaN()
	}
	sort.Float64s(values)
	return values[int(math.Ceil(p*float64(len(values))))-1]
}

func finite(value float64) any {
	if math.IsInf(value, 1) {
		return "INF"
	}
	if math.IsNaN(value) {
		return nil
	}
	return value
}

func main() {
	if runtime.GOOS != "linux" {
		panic("CLOCK_MONOTONIC replay must run in Linux")
	}
	plan := flag.String("schedule", "", "immutable request_id,offset_ns CSV")
	target := flag.String("target", "", "HTTP /work URL")
	out := flag.String("out", "", "request CSV (written after replay)")
	duration := flag.Float64("duration", 0, "measurement seconds")
	timeout := flag.Float64("timeout-ms", 189.2, "planned-time deadline, milliseconds")
	slo := flag.Float64("slo-ms", 18.924, "window p95 limit")
	flag.Parse()
	if *duration <= 0 || *timeout <= 0 || *target == "" || *out == "" || *duration/5 != math.Trunc(*duration/5) {
		panic("invalid replay arguments")
	}
	durationNS := int64(*duration * 1e9)
	records, hash, err := readPlan(*plan, durationNS)
	if err != nil {
		panic(err)
	}
	dialer := &net.Dialer{Timeout: time.Second, KeepAlive: 30 * time.Second}
	tr := &http.Transport{MaxIdleConns: 512, MaxIdleConnsPerHost: 512, DisableCompression: true,
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			conn, err := dialer.DialContext(ctx, network, address)
			if err != nil {
				return nil, err
			}
			return &ackConn{conn.(*net.TCPConn)}, nil
		}}
	client := &http.Client{Transport: tr, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	defer tr.CloseIdleConnections()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var outputMu sync.Mutex
	encoder := json.NewEncoder(os.Stdout)
	emit := func(v any) {
		outputMu.Lock()
		defer outputMu.Unlock()
		if e := encoder.Encode(v); e != nil {
			panic(e)
		}
	}
	// Python's Linux time.monotonic_ns and this CLOCK_MONOTONIC share a domain.
	origin := monoNS() + int64(2*time.Second)
	emit(map[string]any{"type": "start", "start_mono_ns": origin, "schedule_sha256": hash, "planned_requests": len(records), "duration_seconds": *duration, "clock": "Linux CLOCK_MONOTONIC", "write_timestamp": "httptrace.WroteRequest"})
	var boundaries sync.WaitGroup
	boundaries.Add(1)
	go func() {
		defer boundaries.Done()
		for i := 0; i <= int(*duration/5); i++ {
			deadline := origin + int64(i)*int64(5*time.Second)
			if wait := deadline - monoNS(); wait > 0 {
				time.Sleep(time.Duration(wait))
			}
			emit(map[string]any{"type": "boundary", "tick": i, "planned_mono_ns": deadline, "observed_mono_ns": monoNS()})
		}
	}()
	var requests sync.WaitGroup
	cpuStart := cpuSeconds()
	for i := range records {
		deadline := origin + records[i].Offset
		if wait := deadline - monoNS(); wait > 0 {
			time.Sleep(time.Duration(wait))
		}
		requests.Add(1)
		go func(index int) {
			defer requests.Done()
			rec := &records[index]
			remaining := origin + rec.Offset + int64(*timeout*1e6) - monoNS()
			if remaining <= 0 {
				rec.Error = "late_before_write"
				rec.Done = monoNS() - origin
				return
			}
			reqctx, finish := context.WithTimeout(ctx, time.Duration(remaining))
			defer finish()
			trace := &httptrace.ClientTrace{WroteRequest: func(info httptrace.WroteRequestInfo) {
				atomic.AddInt64(&rec.WriteCallbacks, 1)
				if info.Err == nil {
					atomic.CompareAndSwapInt64(&rec.Written, -1, monoNS()-origin)
				}
			}}
			req, e := http.NewRequestWithContext(httptrace.WithClientTrace(reqctx, trace), "GET", *target, nil)
			if e != nil {
				rec.Error = e.Error()
				rec.Done = monoNS() - origin
				return
			}
			resp, e := client.Do(req)
			if e == nil {
				rec.Code = resp.StatusCode
				_, e = io.Copy(io.Discard, resp.Body)
				closeErr := resp.Body.Close()
				if e == nil {
					e = closeErr
				}
				if e == nil && (rec.Code < 200 || rec.Code >= 300) {
					e = fmt.Errorf("HTTP_%d", rec.Code)
				}
			}
			rec.Done = monoNS() - origin
			if e != nil {
				rec.Error = e.Error()
			}
			if rec.Done-rec.Offset > int64(*timeout*1e6) && rec.Error == "" {
				rec.Error = "planned_deadline_exceeded"
			}
		}(i)
	}
	if wait := origin + durationNS + int64(time.Second) - monoNS(); wait > 0 {
		time.Sleep(time.Duration(wait))
	}
	cancel()
	requests.Wait()
	boundaries.Wait()
	cpu := cpuSeconds() - cpuStart
	late, latency := []float64{}, []float64{}
	window := make([][]float64, int(*duration/5))
	errorsN, written, retries := 0, 0, 0
	f, err := os.Create(*out)
	if err != nil {
		panic(err)
	}
	w := csv.NewWriter(f)
	w.Write([]string{"request_id", "offset_ns", "written_ns", "completed_ns", "http_status", "error", "write_callbacks"})
	for i := range records {
		r := &records[i]
		writtenAt := atomic.LoadInt64(&r.Written)
		callbacks := atomic.LoadInt64(&r.WriteCallbacks)
		if callbacks > 1 {
			retries++
		}
		v := math.Inf(1)
		if writtenAt >= 0 {
			written++
			late = append(late, float64(writtenAt-r.Offset)/1e6)
		}
		if r.Error == "" && writtenAt >= 0 {
			v = float64(r.Done-r.Offset) / 1e6
		} else {
			errorsN++
		}
		latency = append(latency, v)
		window[r.Offset/int64(5*time.Second)] = append(window[r.Offset/int64(5*time.Second)], v)
		w.Write([]string{strconv.Itoa(i), strconv.FormatInt(r.Offset, 10), strconv.FormatInt(writtenAt, 10), strconv.FormatInt(r.Done, 10), strconv.Itoa(r.Code), r.Error, strconv.FormatInt(callbacks, 10)})
	}
	w.Flush()
	if err = w.Error(); err != nil {
		panic(err)
	}
	if err = f.Close(); err != nil {
		panic(err)
	}
	windowRows := make([]map[string]any, len(window))
	empty, bad := 0, 0
	for i, values := range window {
		p95 := quantile(values, .95)
		violation := len(values) > 0 && p95 > *slo
		if len(values) == 0 {
			empty++
		}
		if violation {
			bad++
		}
		windowRows[i] = map[string]any{"tick": i, "planned_arrivals": len(values), "p95_ms": finite(p95), "slo_violation": violation}
	}
	emit(map[string]any{"type": "complete", "schedule_sha256": hash, "planned": len(records), "written": written, "missing_write": len(records) - written, "retried_requests": retries, "failed": errorsN, "late_p99_ms": finite(quantile(late, .99)), "latency_p95_ms": finite(quantile(latency, .95)), "sender_cpu_seconds": cpu, "sender_bound_cores": 2, "sender_bound_cpu_fraction": cpu / (*duration * 2), "empty_windows": empty, "slo_violation_windows": bad, "windows": windowRows})
}
