// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package mlog

import (
	"bytes"
	"io"
	"math/rand"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/milvus-io/milvus/pkg/v3/metrics"
)

func TestAsyncBufferedWriteSyncer(t *testing.T) {
	blockWriter := &blockWriter{
		Writer: os.Stdout,
	}
	syncer := NewAsyncTextIOCore(
		&Config{
			AsyncWriteEnable:         true,
			AsyncWriteFlushInterval:  1 * time.Second,
			AsyncWriteDroppedTimeout: 100 * time.Millisecond,
			AsyncWriteStopTimeout:    1 * time.Second,
			AsyncWritePendingLength:  100,
			AsyncWriteBufferSize:     5,
			AsyncWriteMaxBytesPerLog: 2,
		},
		zapcore.AddSync(blockWriter),
		zap.DebugLevel,
	)
	wg := &sync.WaitGroup{}
	for i := 0; i < 1000; i++ {
		wg.Add(1)
		go func() {
			syncer.Write(zapcore.Entry{
				Level:   zap.DebugLevel,
				Message: "test",
			}, []zapcore.Field{
				zap.String("test", "test"),
				zap.Int("test", 1),
			})
			wg.Done()
		}()
	}
	syncer.Sync()
	wg.Wait()
	syncer.Stop()
}

type blockWriter struct {
	io.Writer
}

func (s *blockWriter) Write(p []byte) (n int, err error) {
	time.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)
	if rand.Intn(10) == 0 {
		return 0, errors.New("write error")
	}
	return s.Writer.Write(p)
}

func (s *blockWriter) Sync() error {
	time.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)
	if rand.Intn(10) == 0 {
		return errors.New("sync error")
	}
	return nil
}

func TestAsyncBufferedWriteSyncerHighLevelWaitsLongerForSpace(t *testing.T) {
	writer := newBlockingWriteSyncer()
	defer writer.unblock()
	syncer := newAsyncTextIOCoreForBlockedWriter(writer, 20*time.Millisecond, 500*time.Millisecond)

	require.NoError(t, syncer.Write(zapcore.Entry{Level: zap.InfoLevel, Message: "writing"}, nil))
	writer.waitUntilBlocked(t)
	require.NoError(t, syncer.Write(zapcore.Entry{Level: zap.InfoLevel, Message: "pending"}, nil))

	droppedBefore := testutil.ToFloat64(metrics.LoggingDroppedWriteTotal)
	done := make(chan error, 1)
	go func() {
		done <- syncer.Write(zapcore.Entry{Level: zap.ErrorLevel, Message: "important"}, nil)
	}()

	// An ordinary entry gives up after droppedTimeout, while the Error entry
	// continues waiting for space rather than evicting the pending entry.
	require.NoError(t, syncer.Write(zapcore.Entry{Level: zap.InfoLevel, Message: "droppable"}, nil))
	assert.Equal(t, droppedBefore+1, testutil.ToFloat64(metrics.LoggingDroppedWriteTotal))
	select {
	case <-done:
		require.FailNow(t, "error log returned before queue space became available")
	default:
	}

	writer.unblock()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(time.Second):
		require.FailNow(t, "error log did not resume after queue space became available")
	}
	assert.Equal(t, droppedBefore+1, testutil.ToFloat64(metrics.LoggingDroppedWriteTotal))
	syncer.Stop()
}

func TestAsyncBufferedWriteSyncerHighLevelWriteIsBounded(t *testing.T) {
	writer := newBlockingWriteSyncer()
	defer writer.unblock()
	syncer := newAsyncTextIOCoreForBlockedWriter(writer, 10*time.Millisecond, 50*time.Millisecond)

	require.NoError(t, syncer.Write(zapcore.Entry{Level: zap.InfoLevel, Message: "writing"}, nil))
	writer.waitUntilBlocked(t)
	require.NoError(t, syncer.Write(zapcore.Entry{Level: zap.InfoLevel, Message: "pending"}, nil))

	droppedBefore := testutil.ToFloat64(metrics.LoggingDroppedWriteTotal)
	done := make(chan error, 1)
	go func() {
		done <- syncer.Write(zapcore.Entry{Level: zap.ErrorLevel, Message: "important"}, nil)
	}()
	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(time.Second):
		require.FailNow(t, "error log blocked indefinitely on a full pending queue")
	}
	assert.Equal(t, droppedBefore+1, testutil.ToFloat64(metrics.LoggingDroppedWriteTotal))
	pending := <-syncer.pending
	assert.Contains(t, pending.buf.String(), "pending")
	syncer.pending <- pending

	writer.unblock()
	syncer.Stop()
}

func TestAsyncBufferedWriteSyncerStopTimesOutOnBlockedWriter(t *testing.T) {
	writer := newBlockingWriteSyncer()
	defer writer.unblock()
	syncer := newAsyncTextIOCoreForBlockedWriter(writer, 10*time.Millisecond, 50*time.Millisecond)

	require.NoError(t, syncer.Write(zapcore.Entry{Level: zap.InfoLevel, Message: "writing"}, nil))
	writer.waitUntilBlocked(t)

	done := make(chan struct{})
	go func() {
		syncer.Stop()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		require.FailNow(t, "async logger stop blocked on the underlying writer")
	}

	writer.unblock()
	select {
	case <-syncer.notifier.FinishChan():
	case <-time.After(time.Second):
		require.FailNow(t, "async logger did not finish after the writer was unblocked")
	}
}

func newAsyncTextIOCoreForBlockedWriter(writer zapcore.WriteSyncer, droppedTimeout, stopTimeout time.Duration) *asyncTextIOCore {
	return NewAsyncTextIOCore(
		&Config{
			Format:                      "text",
			AsyncWriteFlushInterval:     time.Hour,
			AsyncWriteDroppedTimeout:    droppedTimeout,
			AsyncWriteNonDroppableLevel: zap.ErrorLevel.String(),
			AsyncWriteStopTimeout:       stopTimeout,
			AsyncWritePendingLength:     1,
			AsyncWriteBufferSize:        1,
			AsyncWriteMaxBytesPerLog:    1024,
		},
		writer,
		zap.DebugLevel,
	)
}

type blockingWriteSyncer struct {
	started     chan struct{}
	release     chan struct{}
	startedOnce sync.Once
	releaseOnce sync.Once
}

func newBlockingWriteSyncer() *blockingWriteSyncer {
	return &blockingWriteSyncer{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
}

func (s *blockingWriteSyncer) Write(p []byte) (int, error) {
	s.startedOnce.Do(func() { close(s.started) })
	<-s.release
	return len(p), nil
}

func (s *blockingWriteSyncer) Sync() error {
	return nil
}

func (s *blockingWriteSyncer) waitUntilBlocked(t *testing.T) {
	t.Helper()
	select {
	case <-s.started:
	case <-time.After(time.Second):
		require.FailNow(t, "writer was not called")
	}
}

func (s *blockingWriteSyncer) unblock() {
	s.releaseOnce.Do(func() { close(s.release) })
}

// slowRecordWriter records everything written to it, slowly, so that entries pile up in the async queue.
type slowRecordWriter struct {
	delay time.Duration
	block chan struct{} // when not nil, Write blocks until it is closed.

	mu  sync.Mutex
	buf bytes.Buffer
}

func (w *slowRecordWriter) Write(p []byte) (int, error) {
	if w.block != nil {
		<-w.block
	}
	time.Sleep(w.delay)
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.Write(p)
}

func (w *slowRecordWriter) Sync() error { return nil }

// noExitHook replaces the os.Exit(1) that zap runs after a Fatal entry.
type noExitHook struct{}

func (noExitHook) OnWrite(*zapcore.CheckedEntry, []zapcore.Field) {}

func (w *slowRecordWriter) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.String()
}

func newTestAsyncCore(ws zapcore.WriteSyncer, stopTimeout time.Duration) *asyncTextIOCore {
	return NewAsyncTextIOCore(
		&Config{
			AsyncWriteEnable:            true,
			AsyncWriteFlushInterval:     time.Hour,
			AsyncWriteDroppedTimeout:    100 * time.Millisecond,
			AsyncWriteNonDroppableLevel: "error",
			AsyncWriteStopTimeout:       stopTimeout,
			AsyncWritePendingLength:     1024,
			AsyncWriteBufferSize:        1, // every entry goes straight to the slow writer.
			AsyncWriteMaxBytesPerLog:    1024,
		},
		ws,
		zap.DebugLevel,
	)
}

// A Fatal entry must be on the underlying writer when the call returns: the process exits right after.
func TestAsyncTextIOCoreFatalIsWrittenBeforeReturn(t *testing.T) {
	w := &slowRecordWriter{delay: 10 * time.Millisecond}
	core := newTestAsyncCore(w, 10*time.Second)
	defer core.Stop()
	// Log the Fatal entry like production does, but do not exit the test process.
	logger := zap.New(core, zap.WithFatalHook(noExitHook{}))

	for i := 0; i < 50; i++ {
		logger.Info("backlog entry")
	}
	logger.Fatal("fatal entry")

	out := w.String()
	assert.Equal(t, 50, strings.Count(out, "backlog entry"), "entries queued before the Fatal must be written first")
	assert.Contains(t, out, "fatal entry")
}

func TestAsyncTextIOCoreSyncWaitsForQueuedEntries(t *testing.T) {
	w := &slowRecordWriter{delay: 10 * time.Millisecond}
	core := newTestAsyncCore(w, 10*time.Second)
	defer core.Stop()
	logger := zap.New(core)

	for i := 0; i < 20; i++ {
		logger.Info("queued entry")
	}
	require.NoError(t, core.Sync())
	assert.Equal(t, 20, strings.Count(w.String(), "queued entry"))
}

func TestAsyncTextIOCoreSyncTimesOutOnBlockedWriter(t *testing.T) {
	w := &slowRecordWriter{block: make(chan struct{})}
	core := newTestAsyncCore(w, 200*time.Millisecond)
	logger := zap.New(core)
	logger.Info("stuck entry")

	start := time.Now()
	err := core.Sync()
	assert.ErrorIs(t, err, errSyncTimeout)
	assert.Less(t, time.Since(start), 5*time.Second)

	close(w.block)
	core.Stop()
}
