//! Request-boundary priority. Active requests are never preempted.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use tokio::sync::Notify;

use super::gpu_gate::{GpuGate, InferencePermit};

pub const PRIORITY_HEADER: &str = "x-inference-priority";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Interactive,
    Normal,
    Background,
}

impl Priority {
    pub fn from_header(value: Option<&str>) -> Self {
        match value {
            Some("interactive") => Self::Interactive,
            Some("background") => Self::Background,
            _ => Self::Normal,
        }
    }
}

#[derive(Default)]
pub struct RequestQueue {
    models: Mutex<HashMap<String, Weak<ModelQueue>>>,
}

#[derive(Default)]
struct ModelQueue {
    state: Mutex<QueueState>,
    changed: Notify,
}

#[derive(Default)]
struct QueueState {
    next_id: u64,
    active: usize,
    active_background: usize,
    waiting: Vec<(Priority, u64)>,
}

struct Ticket {
    queue: Arc<ModelQueue>,
    id: u64,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.queue
            .state
            .lock()
            .unwrap()
            .waiting
            .retain(|(_, id)| *id != self.id);
        self.queue.changed.notify_waiters();
    }
}

pub struct RequestPermit {
    queue: Arc<ModelQueue>,
    pub gpu: Option<InferencePermit>,
    background: bool,
}

impl Drop for RequestPermit {
    fn drop(&mut self) {
        // Release the GPU reader before waking requests behind a media writer.
        self.gpu.take();
        let mut state = self.queue.state.lock().unwrap();
        state.active -= 1;
        if self.background {
            state.active_background -= 1;
        }
        drop(state);
        self.queue.changed.notify_waiters();
    }
}

impl RequestQueue {
    /// Aggregate queue depth only; request bodies are never retained here.
    pub fn queued(&self, model: &str) -> usize {
        self.models
            .lock()
            .unwrap()
            .get(model)
            .and_then(Weak::upgrade)
            .map(|queue| queue.state.lock().unwrap().waiting.len())
            .unwrap_or(0)
    }

    pub async fn acquire(
        &self,
        model: &str,
        priority: Priority,
        capacity: usize,
        gpu: Option<Arc<GpuGate>>,
    ) -> Result<RequestPermit, String> {
        self.acquire_with_background_reserve(model, priority, capacity, 0, gpu)
            .await
    }

    /// Keep capacity available for foreground work instead of filling every
    /// backend slot with potentially long background generations.
    pub async fn acquire_with_background_reserve(
        &self,
        model: &str,
        priority: Priority,
        capacity: usize,
        reserved: usize,
        gpu: Option<Arc<GpuGate>>,
    ) -> Result<RequestPermit, String> {
        let capacity = capacity.max(1);
        let background_capacity = capacity - reserved.min(capacity - 1);
        let queue = {
            let mut models = self.models.lock().unwrap();
            models.retain(|_, queue| queue.strong_count() > 0);
            match models.get(model).and_then(Weak::upgrade) {
                Some(queue) => queue,
                None => {
                    let queue = Arc::new(ModelQueue::default());
                    models.insert(model.into(), Arc::downgrade(&queue));
                    queue
                }
            }
        };
        let ticket = {
            let mut state = queue.state.lock().unwrap();
            let id = state.next_id;
            state.next_id += 1;
            state.waiting.push((priority, id));
            Ticket {
                queue: queue.clone(),
                id,
            }
        };
        queue.changed.notify_waiters();
        loop {
            let changed = queue.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let eligible = {
                let state = queue.state.lock().unwrap();
                state.active < capacity
                    && (priority != Priority::Background
                        || (state.active_background < background_capacity
                            && (reserved == 0 || state.active == state.active_background)))
                    && state.waiting.iter().min() == Some(&(priority, ticket.id))
            };
            if !eligible {
                changed.await;
                continue;
            }
            // A request waiting for media is still queued, so a later chat can
            // overtake it. Queued requests never pin GPU readers or model instances.
            let permit = if let Some(gate) = &gpu {
                tokio::select! {
                    biased;
                    _ = &mut changed => continue,
                    permit = gate.inference() => Some(permit?),
                }
            } else {
                None
            };
            let admitted = {
                let mut state = queue.state.lock().unwrap();
                if state.active < capacity
                    && (priority != Priority::Background
                        || (state.active_background < background_capacity
                            && (reserved == 0 || state.active == state.active_background)))
                    && state.waiting.iter().min() == Some(&(priority, ticket.id))
                {
                    state.waiting.retain(|(_, id)| *id != ticket.id);
                    state.active += 1;
                    if priority == Priority::Background {
                        state.active_background += 1;
                    }
                    true
                } else {
                    false
                }
            };
            if admitted {
                queue.changed.notify_waiters();
                return Ok(RequestPermit {
                    queue: queue.clone(),
                    gpu: permit,
                    background: priority == Priority::Background,
                });
            }
            drop(permit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    async fn waiting(queue: &RequestQueue, model: &str, count: usize) {
        timeout(Duration::from_secs(2), async {
            loop {
                let length = queue
                    .models
                    .lock()
                    .unwrap()
                    .get(model)
                    .and_then(Weak::upgrade)
                    .map(|q| q.state.lock().unwrap().waiting.len())
                    .unwrap_or(0);
                if length == count {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    fn request(
        queue: Arc<RequestQueue>,
        priority: Priority,
        order: Arc<Mutex<Vec<&'static str>>>,
        name: &'static str,
        gpu: Option<Arc<GpuGate>>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let _permit = queue.acquire("model", priority, 1, gpu).await.unwrap();
            order.lock().unwrap().push(name);
        })
    }

    #[tokio::test]
    async fn active_finishes_then_interactive_normal_background_fifo_and_retry_requeues() {
        let queue = Arc::new(RequestQueue::default());
        let first = queue
            .acquire("model", Priority::Background, 1, None)
            .await
            .unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let bg = request(
            queue.clone(),
            Priority::Background,
            order.clone(),
            "background",
            None,
        );
        waiting(&queue, "model", 1).await;
        let normal = request(
            queue.clone(),
            Priority::Normal,
            order.clone(),
            "normal",
            None,
        );
        waiting(&queue, "model", 2).await;
        let chat = request(
            queue.clone(),
            Priority::Interactive,
            order.clone(),
            "chat",
            None,
        );
        waiting(&queue, "model", 3).await;
        let chat2 = request(
            queue.clone(),
            Priority::Interactive,
            order.clone(),
            "chat2",
            None,
        );
        waiting(&queue, "model", 4).await;
        assert!(order.lock().unwrap().is_empty());
        drop(first);
        tokio::try_join!(bg, normal, chat, chat2).unwrap();
        assert_eq!(
            *order.lock().unwrap(),
            vec!["chat", "chat2", "normal", "background"]
        );
        // The original background task retries as a fresh request behind a chat.
        let active = queue
            .acquire("model", Priority::Background, 1, None)
            .await
            .unwrap();
        let chat = request(
            queue.clone(),
            Priority::Interactive,
            order.clone(),
            "next-chat",
            None,
        );
        waiting(&queue, "model", 1).await;
        drop(active);
        let retry = queue
            .acquire("model", Priority::Background, 1, None)
            .await
            .unwrap();
        chat.await.unwrap();
        assert_eq!(order.lock().unwrap().last(), Some(&"next-chat"));
        drop(retry);
    }

    #[tokio::test]
    async fn slots_and_independent_models_continue_and_cancellation_removes_waiters() {
        let queue = Arc::new(RequestQueue::default());
        let first = queue
            .acquire("gpu-model", Priority::Normal, 2, None)
            .await
            .unwrap();
        let second = timeout(
            Duration::from_secs(1),
            queue.acquire("gpu-model", Priority::Normal, 2, None),
        )
        .await
        .unwrap()
        .unwrap();
        let cpu = timeout(
            Duration::from_secs(1),
            queue.acquire("cpu-model", Priority::Background, 1, None),
        )
        .await
        .unwrap()
        .unwrap();
        let q = queue.clone();
        let cancelled =
            tokio::spawn(
                async move { q.acquire("gpu-model", Priority::Interactive, 2, None).await },
            );
        waiting(&queue, "gpu-model", 1).await;
        cancelled.abort();
        let _ = cancelled.await;
        waiting(&queue, "gpu-model", 0).await;
        drop(first);
        let next = timeout(
            Duration::from_secs(1),
            queue.acquire("gpu-model", Priority::Background, 2, None),
        )
        .await
        .unwrap()
        .unwrap();
        drop((second, cpu, next));
    }

    #[tokio::test]
    async fn chats_overtake_background_waiting_for_media_without_pinning_gpu_readers() {
        let queue = Arc::new(RequestQueue::default());
        let gate = Arc::new(GpuGate::default());
        let media = gate.media().await;
        let order = Arc::new(Mutex::new(Vec::new()));
        let bg = request(
            queue.clone(),
            Priority::Background,
            order.clone(),
            "background",
            Some(gate.clone()),
        );
        waiting(&queue, "model", 1).await;
        let chat = request(
            queue.clone(),
            Priority::Interactive,
            order.clone(),
            "chat",
            Some(gate.clone()),
        );
        waiting(&queue, "model", 2).await;
        assert!(order.lock().unwrap().is_empty());
        drop(media);
        tokio::try_join!(bg, chat).unwrap();
        assert_eq!(*order.lock().unwrap(), vec!["chat", "background"]);
        timeout(Duration::from_secs(1), gate.media()).await.unwrap();
    }

    #[tokio::test]
    async fn recovery_failure_drops_queue_ticket() {
        let queue = RequestQueue::default();
        let gate = Arc::new(GpuGate::default());
        gate.set_error(Some("GPU recovering".into()));
        assert!(queue
            .acquire("model", Priority::Interactive, 1, Some(gate.clone()))
            .await
            .is_err());
        gate.set_error(None);
        let _next = timeout(
            Duration::from_secs(1),
            queue.acquire("model", Priority::Normal, 1, Some(gate)),
        )
        .await
        .unwrap()
        .unwrap();
    }

    #[tokio::test]
    async fn reserved_slot_keeps_chat_moving_while_background_generation_continues() {
        let queue = Arc::new(RequestQueue::default());
        let background = queue
            .acquire_with_background_reserve("m", Priority::Background, 2, 1, None)
            .await
            .unwrap();
        let q = queue.clone();
        let mut second = tokio::spawn(async move {
            q.acquire_with_background_reserve("m", Priority::Background, 2, 1, None)
                .await
                .unwrap()
        });
        waiting(&queue, "m", 1).await;
        assert!(!second.is_finished());
        let chat = timeout(
            Duration::from_secs(1),
            queue.acquire_with_background_reserve("m", Priority::Interactive, 2, 1, None),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(
            !second.is_finished(),
            "the active background request keeps its permit; another must wait"
        );
        drop(background);
        assert!(
            timeout(Duration::from_millis(30), &mut second)
                .await
                .is_err(),
            "a new background prefill must wait while foreground generation continues"
        );
        drop(chat);
        let next = timeout(Duration::from_secs(1), second)
            .await
            .unwrap()
            .unwrap();
        drop(next);
        // A one-slot backend still admits background work rather than starving it.
        let _single = queue
            .acquire_with_background_reserve("single", Priority::Background, 1, 1, None)
            .await
            .unwrap();
    }
}
