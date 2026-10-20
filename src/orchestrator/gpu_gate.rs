//! Fair admission for the shared GPU. Inference holds a read permit through
//! loading and the complete response; a media job holds the write permit
//! through draining, generation, cleanup and restoration.
use serde::Serialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

pub type InferencePermit = Arc<OwnedRwLockReadGuard<()>>;

#[derive(Default)]
pub struct GpuGate {
    lock: Arc<RwLock<()>>,
    waiting_inference: AtomicUsize,
    waiting_media: AtomicUsize,
    state: Mutex<GateState>,
}

#[derive(Clone, Serialize)]
pub struct GateState {
    pub phase: String,
    pub media_job: Option<String>,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct GateSnapshot {
    #[serde(flatten)]
    pub state: GateState,
    pub queued_inference: usize,
    pub queued_media: usize,
}

impl Default for GateState {
    fn default() -> Self {
        Self {
            phase: "llm".into(),
            media_job: None,
            error: None,
        }
    }
}

struct Waiting<'a>(&'a AtomicUsize);
impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl GpuGate {
    /// Never jump a queued media writer. A recovery fault also closes GPU admission.
    pub fn try_inference(&self) -> Option<InferencePermit> {
        let permit = self.lock.clone().try_read_owned().ok()?;
        if self.state.lock().unwrap().error.is_some() {
            return None;
        }
        Some(Arc::new(permit))
    }

    pub async fn inference(&self) -> Result<InferencePermit, String> {
        self.waiting_inference.fetch_add(1, Ordering::SeqCst);
        let waiting = Waiting(&self.waiting_inference);
        let permit = self.lock.clone().read_owned().await;
        drop(waiting);
        if let Some(error) = self.state.lock().unwrap().error.clone() {
            return Err(error);
        }
        Ok(Arc::new(permit))
    }

    pub async fn media(&self) -> OwnedRwLockWriteGuard<()> {
        self.waiting_media.fetch_add(1, Ordering::SeqCst);
        let waiting = Waiting(&self.waiting_media);
        let permit = self.lock.clone().write_owned().await;
        drop(waiting);
        permit
    }

    pub fn set_phase(&self, phase: &str, job: Option<&str>) {
        let mut state = self.state.lock().unwrap();
        state.phase = phase.into();
        state.media_job = job.map(str::to_owned);
    }

    pub fn set_error(&self, error: Option<String>) {
        self.state.lock().unwrap().error = error;
    }

    pub fn snapshot(&self) -> GateSnapshot {
        let mut state = self.state.lock().unwrap().clone();
        if state.phase == "llm" && self.waiting_media.load(Ordering::SeqCst) > 0 {
            state.phase = "draining".into();
        }
        GateSnapshot {
            state,
            queued_inference: self.waiting_inference.load(Ordering::SeqCst),
            queued_media: self.waiting_media.load(Ordering::SeqCst),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    #[tokio::test]
    async fn writer_drains_active_readers_and_blocks_all_later_arrivals() {
        let gate = Arc::new(GpuGate::default());
        let active = gate.inference().await.unwrap();
        let media_gate = gate.clone();
        let media = tokio::spawn(async move { media_gate.media().await });
        while gate.snapshot().queued_media == 0 {
            tokio::task::yield_now().await;
        }
        let next_gate = gate.clone();
        let mut next = tokio::spawn(async move { next_gate.inference().await });
        assert!(timeout(Duration::from_millis(20), &mut next).await.is_err());
        assert_eq!(gate.snapshot().queued_inference, 1);
        drop(active);
        let exclusive = media.await.unwrap();
        assert!(timeout(Duration::from_millis(20), &mut next).await.is_err());
        drop(exclusive);
        assert!(next.await.unwrap().is_ok());
        assert_eq!(gate.snapshot().queued_inference, 0);
    }

    #[tokio::test]
    async fn cancellation_removes_waiters_and_readers_run_before_later_media() {
        let gate = Arc::new(GpuGate::default());
        let first = gate.media().await;
        let g = gate.clone();
        let cancelled = tokio::spawn(async move { g.inference().await });
        let g = gate.clone();
        let reader = tokio::spawn(async move { g.inference().await });
        while gate.snapshot().queued_inference < 2 {
            tokio::task::yield_now().await;
        }
        let g = gate.clone();
        let mut second_media = tokio::spawn(async move { g.media().await });
        while gate.snapshot().queued_media == 0 {
            tokio::task::yield_now().await;
        }
        cancelled.abort();
        let _ = cancelled.await;
        assert_eq!(gate.snapshot().queued_inference, 1);
        drop(first);
        let admitted = reader.await.unwrap().unwrap();
        assert!(timeout(Duration::from_millis(20), &mut second_media)
            .await
            .is_err());
        drop(admitted);
        drop(second_media.await.unwrap());
    }
}
