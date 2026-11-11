//! Slot leases last through streamed responses; queued chats do not occupy
//! global inference admission or the dedicated compaction slot.
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use tokio::sync::Notify;

#[derive(Default)]
struct Slot {
    busy: bool,
    keys: Vec<u64>,
    last_used: u64,
}

#[derive(Default)]
struct State {
    slots: HashMap<(String, u32), Slot>,
    clock: u64,
}

#[derive(Default)]
pub struct SlotPool {
    state: Mutex<State>,
    changed: Notify,
}

pub struct SlotLease {
    pool: Arc<SlotPool>,
    model: String,
    pub slot: u32,
}

impl SlotPool {
    pub async fn acquire(
        self: &Arc<Self>,
        model: &str,
        allowed: &[u32],
        keys: &[u64],
    ) -> SlotLease {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                for &slot in allowed {
                    state.slots.entry((model.to_owned(), slot)).or_default();
                }
                let selected = allowed
                    .iter()
                    .copied()
                    .filter(|id| !state.slots[&(model.to_owned(), *id)].busy)
                    .max_by_key(|id| {
                        let slot = &state.slots[&(model.to_owned(), *id)];
                        let matched = slot
                            .keys
                            .iter()
                            .zip(keys)
                            .take_while(|(a, b)| a == b)
                            .count();
                        (matched, std::cmp::Reverse(slot.last_used))
                    });
                if let Some(id) = selected {
                    state.clock += 1;
                    let now = state.clock;
                    let slot = state.slots.get_mut(&(model.to_owned(), id)).unwrap();
                    slot.busy = true;
                    slot.keys = keys.to_vec();
                    slot.last_used = now;
                    return SlotLease {
                        pool: self.clone(),
                        model: model.to_owned(),
                        slot: id,
                    };
                }
            }
            changed.await;
        }
    }
}

impl Drop for SlotLease {
    fn drop(&mut self) {
        let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = state.slots.get_mut(&(self.model.clone(), self.slot)) {
            slot.busy = false;
        }
        drop(state);
        self.pool.changed.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn chat_pool_preserves_affinity_and_leaves_compaction_available() {
        let pool = Arc::new(SlotPool::default());
        let first = pool.acquire("model", &[0, 1], &[11]).await;
        let first_id = first.slot;
        let second = pool.acquire("model", &[0, 1], &[22]).await;
        assert_ne!(first.slot, second.slot);
        let waiting = pool.acquire("model", &[0, 1], &[33]);
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), &mut waiting)
                .await
                .is_err()
        );
        let compaction = pool.acquire("model", &[2], &[]).await;
        assert_eq!(compaction.slot, 2);
        drop(first);
        assert_eq!(waiting.await.slot, first_id);
        drop(second);
        let affinity = pool.acquire("model", &[0, 1], &[22, 44]).await;
        assert_ne!(affinity.slot, first_id);
    }
}
