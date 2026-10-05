//! Dynamic request batching.
//!
//! A [`Batcher`] collects independently submitted items into batches and runs
//! each batch through a [`BatchExecutor`] once. It is used inside the worker
//! subprocess: every prediction still occupies its own slot (and so keeps its
//! own ID, status, webhook, and cancellation), while the batcher groups the
//! model invocations of concurrently running slots into a single call.
//!
//! A batch is dispatched when either:
//! - `max_batch_size` items are queued, or
//! - the oldest queued item has waited `max_wait`.
//!
//! The scheduler is a single tokio task that blocks on its queue (no
//! busy-spinning) and executes at most one batch at a time. Items that arrive
//! while a batch is executing are queued and form the next batch as soon as
//! the executor is free.
//!
//! Cancellation is per item: a cancelled item that has not been dispatched is
//! dropped from the queue; a cancelled item inside a running batch resolves
//! immediately as cancelled while the rest of the batch keeps running. When
//! every item of a running batch is cancelled, the executor's cancel token
//! fires so it can interrupt the model call.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

use crate::prediction::CancellationToken;

/// Batching parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchConfig {
    /// Maximum number of items per batch (>= 1).
    pub max_batch_size: usize,
    /// Maximum time the oldest queued item waits before a partial batch is
    /// dispatched.
    pub max_wait: Duration,
}

impl BatchConfig {
    pub fn new(max_batch_size: usize, max_wait: Duration) -> Self {
        Self {
            max_batch_size: max_batch_size.max(1),
            max_wait,
        }
    }
}

/// Why a batched item did not produce an output.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BatchError {
    /// The executor failed for the whole batch.
    #[error("{0}")]
    Failed(String),
    /// The item was cancelled.
    #[error("cancelled")]
    Cancelled,
    /// The batcher shut down before the item was executed.
    #[error("worker shutting down before the batched prediction ran")]
    Shutdown,
}

/// Execution statistics shared by all items of a batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchStats {
    /// Number of items executed together.
    pub batch_size: usize,
    /// Time this item spent queued before its batch was dispatched.
    pub queue_time: Duration,
    /// Wall time of the executor call.
    pub run_time: Duration,
}

/// The result delivered to one submitted item.
#[derive(Debug)]
pub struct BatchOutcome<O> {
    pub output: Result<O, BatchError>,
    /// None when the item never reached the executor.
    pub stats: Option<BatchStats>,
}

impl<O> BatchOutcome<O> {
    fn unexecuted(error: BatchError) -> Self {
        Self {
            output: Err(error),
            stats: None,
        }
    }
}

/// Runs one batch of inputs.
#[async_trait::async_trait]
pub trait BatchExecutor<I, O>: Send + Sync + 'static
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Execute a batch. Must return exactly one output per input, in order.
    ///
    /// `cancel` fires when every item of the batch has been cancelled; the
    /// executor may use it to interrupt the underlying call.
    async fn execute(&self, inputs: Vec<I>, cancel: CancellationToken) -> Result<Vec<O>, String>;
}

struct QueuedItem<I, O> {
    input: I,
    enqueued_at: Instant,
    cancel: CancellationToken,
    responder: oneshot::Sender<BatchOutcome<O>>,
}

impl<I, O> QueuedItem<I, O> {
    fn is_live(&self) -> bool {
        !self.cancel.is_cancelled() && !self.responder.is_closed()
    }
}

/// Handle to a dynamic batching scheduler.
///
/// Cloning is cheap; all clones submit to the same scheduler.
pub struct Batcher<I, O> {
    tx: mpsc::UnboundedSender<QueuedItem<I, O>>,
    shutdown: CancellationToken,
    config: BatchConfig,
}

impl<I, O> Clone for Batcher<I, O> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            shutdown: self.shutdown.clone(),
            config: self.config,
        }
    }
}

impl<I, O> Batcher<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Spawn the scheduler task on the current tokio runtime.
    pub fn spawn(config: BatchConfig, executor: Arc<dyn BatchExecutor<I, O>>) -> Self {
        let config = BatchConfig::new(config.max_batch_size, config.max_wait);
        let (tx, rx) = mpsc::unbounded_channel();
        let shutdown = CancellationToken::new();
        tokio::spawn(run_scheduler(config, executor, rx, shutdown.clone()));
        Self {
            tx,
            shutdown,
            config,
        }
    }

    pub fn config(&self) -> BatchConfig {
        self.config
    }

    /// Submit one item and wait for its outcome.
    ///
    /// Resolves as [`BatchError::Cancelled`] as soon as `cancel` fires, even if
    /// the item's batch is still running.
    pub async fn submit(&self, input: I, cancel: CancellationToken) -> BatchOutcome<O> {
        if cancel.is_cancelled() {
            return BatchOutcome::unexecuted(BatchError::Cancelled);
        }
        if self.shutdown.is_cancelled() {
            return BatchOutcome::unexecuted(BatchError::Shutdown);
        }
        let (responder, rx) = oneshot::channel();
        let item = QueuedItem {
            input,
            enqueued_at: Instant::now(),
            cancel: cancel.clone(),
            responder,
        };
        if self.tx.send(item).is_err() {
            return BatchOutcome::unexecuted(BatchError::Shutdown);
        }
        tokio::select! {
            biased;
            outcome = rx => outcome.unwrap_or_else(|_| BatchOutcome::unexecuted(BatchError::Shutdown)),
            _ = cancel.cancelled() => BatchOutcome::unexecuted(BatchError::Cancelled),
        }
    }

    /// Stop the scheduler. Queued items that have not been dispatched resolve
    /// as [`BatchError::Shutdown`]; a batch that is already executing runs to
    /// completion.
    pub fn shutdown(&self) {
        self.shutdown.cancel();
    }
}

/// Wait for the next live item, or None when the queue is closed / shut down.
async fn next_live_item<I, O>(
    rx: &mut mpsc::UnboundedReceiver<QueuedItem<I, O>>,
    shutdown: &CancellationToken,
) -> Option<QueuedItem<I, O>> {
    loop {
        let item = tokio::select! {
            biased;
            _ = shutdown.cancelled() => return None,
            item = rx.recv() => item?,
        };
        if item.is_live() {
            return Some(item);
        }
    }
}

async fn run_scheduler<I, O>(
    config: BatchConfig,
    executor: Arc<dyn BatchExecutor<I, O>>,
    mut rx: mpsc::UnboundedReceiver<QueuedItem<I, O>>,
    shutdown: CancellationToken,
) where
    I: Send + 'static,
    O: Send + 'static,
{
    while let Some(first) = next_live_item(&mut rx, &shutdown).await {
        let deadline = first.enqueued_at + config.max_wait;
        let mut batch = vec![first];

        // Fill the batch until it is full or the oldest item's deadline passes.
        // `timeout_at` polls the queue before checking the deadline, so items
        // that queued up while the previous batch was running are collected
        // immediately even though their deadline has already passed.
        let mut shutting_down = false;
        while batch.len() < config.max_batch_size {
            let next = tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    shutting_down = true;
                    break;
                }
                next = tokio::time::timeout_at(deadline, rx.recv()) => next,
            };
            match next {
                Ok(Some(item)) => {
                    if item.is_live() {
                        batch.push(item);
                    }
                }
                // Queue closed: dispatch what we have.
                Ok(None) => break,
                // Deadline reached.
                Err(_) => break,
            }
        }

        if shutting_down {
            fail_all(batch, BatchError::Shutdown);
            break;
        }

        // Items may have been cancelled while the batch was forming.
        batch.retain(QueuedItem::is_live);
        if batch.is_empty() {
            continue;
        }

        execute_batch(&*executor, batch).await;
    }

    // Shutdown or queue closed: fail anything still queued.
    rx.close();
    let mut remaining = Vec::new();
    while let Ok(item) = rx.try_recv() {
        remaining.push(item);
    }
    fail_all(remaining, BatchError::Shutdown);
}

fn fail_all<I, O>(items: Vec<QueuedItem<I, O>>, error: BatchError) {
    for item in items {
        let _ = item.responder.send(BatchOutcome::unexecuted(error.clone()));
    }
}

async fn execute_batch<I, O>(executor: &dyn BatchExecutor<I, O>, batch: Vec<QueuedItem<I, O>>)
where
    I: Send + 'static,
    O: Send + 'static,
{
    let dispatched_at = Instant::now();
    let batch_size = batch.len();
    let mut inputs = Vec::with_capacity(batch_size);
    let mut members = Vec::with_capacity(batch_size);
    for item in batch {
        inputs.push(item.input);
        members.push((item.enqueued_at, item.cancel, item.responder));
    }

    tracing::debug!(batch_size, "Dispatching batch");

    // Fire the executor's cancel token once every member is cancelled.
    let batch_cancel = CancellationToken::new();
    let all_cancelled =
        futures::future::join_all(members.iter().map(|(_, cancel, _)| cancel.cancelled()));
    let exec = executor.execute(inputs, batch_cancel.clone());
    tokio::pin!(exec);
    let result = tokio::select! {
        result = &mut exec => result,
        _ = all_cancelled => {
            tracing::debug!(batch_size, "All batch members cancelled, cancelling batch");
            batch_cancel.cancel();
            (&mut exec).await
        }
    };
    let run_time = dispatched_at.elapsed();

    tracing::debug!(
        batch_size,
        run_time_ms = run_time.as_millis() as u64,
        ok = result.is_ok(),
        "Batch finished"
    );

    let stats_for = |enqueued_at: Instant| BatchStats {
        batch_size,
        queue_time: dispatched_at.saturating_duration_since(enqueued_at),
        run_time,
    };

    match result {
        Ok(outputs) if outputs.len() == batch_size => {
            for ((enqueued_at, cancel, responder), output) in members.into_iter().zip(outputs) {
                let output = if cancel.is_cancelled() {
                    Err(BatchError::Cancelled)
                } else {
                    Ok(output)
                };
                let _ = responder.send(BatchOutcome {
                    output,
                    stats: Some(stats_for(enqueued_at)),
                });
            }
        }
        Ok(outputs) => {
            let error = BatchError::Failed(format!(
                "batch returned {} outputs for a batch of {} inputs",
                outputs.len(),
                batch_size
            ));
            for (enqueued_at, _, responder) in members {
                let _ = responder.send(BatchOutcome {
                    output: Err(error.clone()),
                    stats: Some(stats_for(enqueued_at)),
                });
            }
        }
        Err(message) => {
            for (enqueued_at, cancel, responder) in members {
                let output = if cancel.is_cancelled() {
                    Err(BatchError::Cancelled)
                } else {
                    Err(BatchError::Failed(message.clone()))
                };
                let _ = responder.send(BatchOutcome {
                    output,
                    stats: Some(stats_for(enqueued_at)),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Executor that records each batch and echoes inputs (times 10).
    struct RecordingExecutor {
        batches: Mutex<Vec<Vec<u32>>>,
        delay: Duration,
        in_flight: AtomicUsize,
        max_in_flight: AtomicUsize,
    }

    impl RecordingExecutor {
        fn new(delay: Duration) -> Arc<Self> {
            Arc::new(Self {
                batches: Mutex::new(Vec::new()),
                delay,
                in_flight: AtomicUsize::new(0),
                max_in_flight: AtomicUsize::new(0),
            })
        }

        fn batches(&self) -> Vec<Vec<u32>> {
            self.batches.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl BatchExecutor<u32, u32> for RecordingExecutor {
        async fn execute(
            &self,
            inputs: Vec<u32>,
            cancel: CancellationToken,
        ) -> Result<Vec<u32>, String> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(now, Ordering::SeqCst);
            self.batches.lock().unwrap().push(inputs.clone());
            let cancelled = tokio::select! {
                _ = tokio::time::sleep(self.delay) => false,
                _ = cancel.cancelled() => true,
            };
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            if cancelled {
                return Err("interrupted".to_string());
            }
            Ok(inputs.into_iter().map(|i| i * 10).collect())
        }
    }

    struct FailingExecutor;

    #[async_trait::async_trait]
    impl BatchExecutor<u32, u32> for FailingExecutor {
        async fn execute(&self, _: Vec<u32>, _: CancellationToken) -> Result<Vec<u32>, String> {
            Err("CUDA out of memory".to_string())
        }
    }

    struct ShortExecutor;

    #[async_trait::async_trait]
    impl BatchExecutor<u32, u32> for ShortExecutor {
        async fn execute(
            &self,
            inputs: Vec<u32>,
            _: CancellationToken,
        ) -> Result<Vec<u32>, String> {
            Ok(inputs.into_iter().skip(1).collect())
        }
    }

    fn config(max_batch_size: usize, max_wait_ms: u64) -> BatchConfig {
        BatchConfig::new(max_batch_size, Duration::from_millis(max_wait_ms))
    }

    fn submit(
        batcher: &Batcher<u32, u32>,
        input: u32,
    ) -> tokio::task::JoinHandle<BatchOutcome<u32>> {
        let batcher = batcher.clone();
        tokio::spawn(async move { batcher.submit(input, CancellationToken::new()).await })
    }

    fn submit_with(
        batcher: &Batcher<u32, u32>,
        input: u32,
        cancel: CancellationToken,
    ) -> tokio::task::JoinHandle<BatchOutcome<u32>> {
        let batcher = batcher.clone();
        tokio::spawn(async move { batcher.submit(input, cancel).await })
    }

    /// Let spawned tasks run without advancing the (paused) clock.
    async fn settle() {
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn single_item_dispatches_after_max_wait() {
        let exec = RecordingExecutor::new(Duration::ZERO);
        let batcher = Batcher::spawn(config(8, 50), exec.clone());

        let handle = submit(&batcher, 1);
        settle().await;
        tokio::time::advance(Duration::from_millis(49)).await;
        settle().await;
        assert!(exec.batches().is_empty(), "must wait for max_wait");
        assert!(!handle.is_finished());

        tokio::time::advance(Duration::from_millis(1)).await;
        let outcome = handle.await.unwrap();
        assert_eq!(outcome.output, Ok(10));
        let stats = outcome.stats.unwrap();
        assert_eq!(stats.batch_size, 1);
        assert_eq!(stats.queue_time, Duration::from_millis(50));
        assert_eq!(exec.batches(), vec![vec![1]]);
    }

    #[tokio::test(start_paused = true)]
    async fn full_batch_dispatches_without_waiting() {
        let exec = RecordingExecutor::new(Duration::ZERO);
        let batcher = Batcher::spawn(config(3, 10_000), exec.clone());

        let handles: Vec<_> = (1..=3).map(|i| submit(&batcher, i)).collect();
        settle().await;
        // No time has passed: the batch was dispatched because it filled up.
        assert_eq!(exec.batches(), vec![vec![1, 2, 3]]);
        for (i, handle) in handles.into_iter().enumerate() {
            let outcome = handle.await.unwrap();
            assert_eq!(outcome.output, Ok((i as u32 + 1) * 10));
            let stats = outcome.stats.unwrap();
            assert_eq!(stats.batch_size, 3);
            assert_eq!(stats.queue_time, Duration::ZERO);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn partial_batch_dispatches_at_deadline_of_oldest_item() {
        let exec = RecordingExecutor::new(Duration::ZERO);
        let batcher = Batcher::spawn(config(8, 20), exec.clone());

        let a = submit(&batcher, 1);
        settle().await;
        tokio::time::advance(Duration::from_millis(15)).await;
        let b = submit(&batcher, 2);
        settle().await;
        assert!(exec.batches().is_empty());

        // Deadline is measured from the oldest item (t=20ms), not the newest.
        tokio::time::advance(Duration::from_millis(5)).await;
        settle().await;
        assert_eq!(exec.batches(), vec![vec![1, 2]]);
        let a = a.await.unwrap().stats.unwrap();
        let b = b.await.unwrap().stats.unwrap();
        assert_eq!(a.queue_time, Duration::from_millis(20));
        assert_eq!(b.queue_time, Duration::from_millis(5));
    }

    #[tokio::test(start_paused = true)]
    async fn overflow_forms_consecutive_batches() {
        let exec = RecordingExecutor::new(Duration::from_millis(100));
        let batcher = Batcher::spawn(config(2, 5), exec.clone());

        let handles: Vec<_> = (1..=5).map(|i| submit(&batcher, i)).collect();
        for (i, handle) in handles.into_iter().enumerate() {
            assert_eq!(handle.await.unwrap().output, Ok((i as u32 + 1) * 10));
        }
        assert_eq!(exec.batches(), vec![vec![1, 2], vec![3, 4], vec![5]]);
        assert_eq!(
            exec.max_in_flight.load(Ordering::SeqCst),
            1,
            "batches must execute one at a time"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn items_queued_during_execution_dispatch_immediately_after() {
        let exec = RecordingExecutor::new(Duration::from_millis(100));
        let batcher = Batcher::spawn(config(4, 5), exec.clone());

        let first = submit(&batcher, 1);
        settle().await;
        tokio::time::advance(Duration::from_millis(5)).await;
        settle().await;
        assert_eq!(exec.batches(), vec![vec![1]]);

        // Arrive while batch 1 is executing.
        let second: Vec<_> = (2..=3).map(|i| submit(&batcher, i)).collect();
        settle().await;
        tokio::time::advance(Duration::from_millis(100)).await;
        settle().await;
        // Their deadline already passed, so they dispatch as soon as the
        // executor frees up -- no extra max_wait.
        assert_eq!(exec.batches(), vec![vec![1], vec![2, 3]]);
        assert_eq!(first.await.unwrap().output, Ok(10));
        for handle in second {
            let stats = handle.await.unwrap().stats.unwrap();
            assert_eq!(stats.batch_size, 2);
            assert_eq!(stats.queue_time, Duration::from_millis(100));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn zero_wait_dispatches_immediately() {
        let exec = RecordingExecutor::new(Duration::ZERO);
        let batcher = Batcher::spawn(config(8, 0), exec.clone());
        let outcome = submit(&batcher, 7).await.unwrap();
        assert_eq!(outcome.output, Ok(70));
    }

    #[tokio::test(start_paused = true)]
    async fn outputs_are_routed_to_their_submitters() {
        let exec = RecordingExecutor::new(Duration::ZERO);
        let batcher = Batcher::spawn(config(16, 5), exec.clone());
        let handles: Vec<_> = (0..16).rev().map(|i| (i, submit(&batcher, i))).collect();
        for (input, handle) in handles {
            assert_eq!(handle.await.unwrap().output, Ok(input * 10));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_before_dispatch_removes_item_from_batch() {
        let exec = RecordingExecutor::new(Duration::ZERO);
        let batcher = Batcher::spawn(config(8, 50), exec.clone());

        let cancel = CancellationToken::new();
        let a = submit(&batcher, 1);
        let b = submit_with(&batcher, 2, cancel.clone());
        let c = submit(&batcher, 3);
        settle().await;
        cancel.cancel();

        let b = b.await.unwrap();
        assert_eq!(b.output, Err(BatchError::Cancelled));
        assert!(b.stats.is_none(), "cancelled item never ran");

        tokio::time::advance(Duration::from_millis(50)).await;
        assert_eq!(a.await.unwrap().output, Ok(10));
        assert_eq!(c.await.unwrap().output, Ok(30));
        assert_eq!(exec.batches(), vec![vec![1, 3]]);
    }

    #[tokio::test(start_paused = true)]
    async fn cancelling_only_queued_item_skips_execution() {
        let exec = RecordingExecutor::new(Duration::ZERO);
        let batcher = Batcher::spawn(config(8, 50), exec.clone());

        let cancel = CancellationToken::new();
        let a = submit_with(&batcher, 1, cancel.clone());
        settle().await;
        cancel.cancel();
        assert_eq!(a.await.unwrap().output, Err(BatchError::Cancelled));
        tokio::time::advance(Duration::from_millis(100)).await;
        settle().await;
        assert!(exec.batches().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn already_cancelled_item_is_rejected() {
        let exec = RecordingExecutor::new(Duration::ZERO);
        let batcher = Batcher::spawn(config(8, 0), exec.clone());
        let cancel = CancellationToken::new();
        cancel.cancel();
        let outcome = batcher.submit(1, cancel).await;
        assert_eq!(outcome.output, Err(BatchError::Cancelled));
        settle().await;
        assert!(exec.batches().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_during_execution_resolves_only_that_item() {
        let exec = RecordingExecutor::new(Duration::from_millis(100));
        let batcher = Batcher::spawn(config(2, 1_000), exec.clone());

        let cancel = CancellationToken::new();
        let a = submit_with(&batcher, 1, cancel.clone());
        let b = submit(&batcher, 2);
        settle().await;
        assert_eq!(exec.batches(), vec![vec![1, 2]]);

        tokio::time::advance(Duration::from_millis(10)).await;
        cancel.cancel();
        // The cancelled item resolves without waiting for the batch.
        assert_eq!(a.await.unwrap().output, Err(BatchError::Cancelled));
        assert!(!b.is_finished());

        tokio::time::advance(Duration::from_millis(90)).await;
        assert_eq!(b.await.unwrap().output, Ok(20));
    }

    #[tokio::test(start_paused = true)]
    async fn cancelling_every_member_cancels_the_executor() {
        let exec = RecordingExecutor::new(Duration::from_secs(3600));
        let batcher = Batcher::spawn(config(2, 1_000), exec.clone());

        let ca = CancellationToken::new();
        let cb = CancellationToken::new();
        let a = submit_with(&batcher, 1, ca.clone());
        let b = submit_with(&batcher, 2, cb.clone());
        settle().await;
        assert_eq!(exec.in_flight.load(Ordering::SeqCst), 1);

        ca.cancel();
        settle().await;
        assert_eq!(
            exec.in_flight.load(Ordering::SeqCst),
            1,
            "one member still live"
        );
        cb.cancel();
        settle().await;
        assert_eq!(
            exec.in_flight.load(Ordering::SeqCst),
            0,
            "executor interrupted"
        );
        assert_eq!(a.await.unwrap().output, Err(BatchError::Cancelled));
        assert_eq!(b.await.unwrap().output, Err(BatchError::Cancelled));

        // The scheduler is free for the next batch.
        let c = submit(&batcher, 3);
        settle().await;
        tokio::time::advance(Duration::from_millis(1_000)).await;
        settle().await;
        assert_eq!(exec.batches(), vec![vec![1, 2], vec![3]]);
        drop(c);
    }

    #[tokio::test(start_paused = true)]
    async fn executor_error_fails_every_member() {
        let batcher = Batcher::spawn(config(2, 5), Arc::new(FailingExecutor));
        let a = submit(&batcher, 1);
        let b = submit(&batcher, 2);
        for handle in [a, b] {
            let outcome = handle.await.unwrap();
            assert_eq!(
                outcome.output,
                Err(BatchError::Failed("CUDA out of memory".to_string()))
            );
            assert_eq!(outcome.stats.unwrap().batch_size, 2);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn wrong_output_count_fails_every_member() {
        let batcher = Batcher::spawn(config(3, 5), Arc::new(ShortExecutor));
        let handles: Vec<_> = (1..=3).map(|i| submit(&batcher, i)).collect();
        for handle in handles {
            assert_eq!(
                handle.await.unwrap().output,
                Err(BatchError::Failed(
                    "batch returned 2 outputs for a batch of 3 inputs".to_string()
                ))
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn scheduler_survives_failed_batch() {
        struct FlakyExecutor(AtomicUsize);

        #[async_trait::async_trait]
        impl BatchExecutor<u32, u32> for FlakyExecutor {
            async fn execute(
                &self,
                inputs: Vec<u32>,
                _: CancellationToken,
            ) -> Result<Vec<u32>, String> {
                if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err("first batch fails".to_string());
                }
                Ok(inputs)
            }
        }

        let batcher = Batcher::spawn(config(1, 0), Arc::new(FlakyExecutor(AtomicUsize::new(0))));
        assert!(submit(&batcher, 1).await.unwrap().output.is_err());
        assert_eq!(submit(&batcher, 2).await.unwrap().output, Ok(2));
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_fails_queued_items() {
        let exec = RecordingExecutor::new(Duration::ZERO);
        let batcher = Batcher::spawn(config(8, 10_000), exec.clone());

        let a = submit(&batcher, 1);
        let b = submit(&batcher, 2);
        settle().await;
        batcher.shutdown();

        assert_eq!(a.await.unwrap().output, Err(BatchError::Shutdown));
        assert_eq!(b.await.unwrap().output, Err(BatchError::Shutdown));
        assert!(exec.batches().is_empty());

        // New submissions are rejected after shutdown.
        let late = batcher.submit(3, CancellationToken::new()).await;
        assert_eq!(late.output, Err(BatchError::Shutdown));
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_lets_running_batch_finish() {
        let exec = RecordingExecutor::new(Duration::from_millis(100));
        let batcher = Batcher::spawn(config(1, 0), exec.clone());

        let running = submit(&batcher, 1);
        settle().await;
        assert_eq!(exec.batches(), vec![vec![1]]);
        let queued = submit(&batcher, 2);
        settle().await;
        batcher.shutdown();

        tokio::time::advance(Duration::from_millis(100)).await;
        assert_eq!(running.await.unwrap().output, Ok(10));
        assert_eq!(queued.await.unwrap().output, Err(BatchError::Shutdown));
        assert_eq!(exec.batches(), vec![vec![1]]);
    }

    #[test]
    fn config_clamps_batch_size_to_one() {
        assert_eq!(BatchConfig::new(0, Duration::ZERO).max_batch_size, 1);
    }
}
