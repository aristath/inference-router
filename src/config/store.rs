use serde::{de::DeserializeOwned, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::debug;

static SAVE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A generic file-backed JSON store.
///
/// Provides atomic load/save (write-to-temp-then-rename) and snapshot /
/// replace semantics. The internal `std::sync::Mutex` never escapes this
/// module, and the public API only returns owned values, so callers cannot
/// accidentally hold the lock across an `await` and block the tokio
/// runtime.
pub struct JsonStore<T: Clone + DeserializeOwned + Serialize + Send + Sync> {
    path: PathBuf,
    data: std::sync::Mutex<T>,
    save_lock: std::sync::Mutex<()>,
}

impl<T: Clone + DeserializeOwned + Serialize + Send + Sync> JsonStore<T> {
    /// Creates a new store at the given path, loading existing data if present.
    #[allow(dead_code)]
    pub fn new(path: PathBuf) -> Self {
        Self::try_new(path).expect("failed to initialize JSON store")
    }

    /// Creates a new store, returning load errors for existing-but-invalid files.
    pub fn try_new(path: PathBuf) -> Result<Self, StoreError> {
        let data = match Self::load_file(&path) {
            Ok(data) => {
                debug!("Loaded config from {}", path.display());
                data
            }
            Err(StoreError::NotFound(_)) => {
                debug!(
                    "No existing config at {}, will create on first save",
                    path.display(),
                );
                Self::empty_data()
            }
            Err(e) => return Err(e),
        };
        Ok(Self {
            path,
            data: std::sync::Mutex::new(data),
            save_lock: std::sync::Mutex::new(()),
        })
    }

    /// Returns a cloned snapshot of the current data. Never holds the
    /// underlying mutex across the caller's code paths.
    pub fn snapshot(&self) -> T {
        self.data.lock().expect("JsonStore lock poisoned").clone()
    }

    /// Replaces the in-memory data with `new`. Paired with `snapshot` for
    /// read-modify-write on the caller's owned value.
    pub fn replace(&self, new: T) {
        *self.data.lock().expect("JsonStore lock poisoned") = new;
    }

    /// Mutates the in-memory data under the lock via a short closure. The
    /// closure must NOT await, call back into this store, or block — it
    /// runs while the synchronous mutex is held.
    pub fn with_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        let mut guard = self.data.lock().expect("JsonStore lock poisoned");
        f(&mut guard)
    }

    /// Saves the current data to disk atomically.
    pub fn save(&self) -> Result<(), StoreError> {
        let _save = self.save_lock.lock().expect("JsonStore save lock poisoned");
        let snapshot = self.snapshot();
        let json = serde_json::to_string_pretty(&snapshot).map_err(StoreError::Serialization)?;

        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(StoreError::Io)?;
        }

        let temp_path = self.temp_path();
        let write_result = (|| {
            let mut file = fs::File::create(&temp_path)?;
            file.write_all(json.as_bytes()).map_err(StoreError::Io)?;
            file.sync_all().map_err(StoreError::Io)?;
            Ok::<(), StoreError>(())
        })();
        if let Err(e) = write_result {
            let _ = fs::remove_file(&temp_path);
            return Err(e);
        }
        if let Err(e) = fs::rename(&temp_path, &self.path).map_err(StoreError::Io) {
            let _ = fs::remove_file(&temp_path);
            return Err(e);
        }
        sync_parent_dir(&self.path)?;

        debug!("Saved config to {}", self.path.display());
        Ok(())
    }

    fn temp_path(&self) -> PathBuf {
        let seq = SAVE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let file_name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("store.json");
        self.path
            .with_file_name(format!("{file_name}.{pid}.{seq}.tmp"))
    }

    /// Loads a single file from disk.
    fn load_file(path: &Path) -> Result<T, StoreError> {
        if !path.exists() {
            return Err(StoreError::NotFound(path.to_path_buf()));
        }
        let contents = fs::read_to_string(path).map_err(StoreError::Io)?;
        serde_json::from_str(&contents).map_err(StoreError::Deserialization)
    }

    /// Returns the default empty data.
    fn empty_data() -> T {
        // We don't require T: Default; most of our stores use Vec<T>, so an
        // empty JSON array deserializes cleanly. Fall back to an empty JSON
        // object for map-shaped configs.
        serde_json::from_value(serde_json::Value::Array(vec![])).unwrap_or_else(|_| {
            serde_json::from_value(serde_json::Value::Object(serde_json::Map::new()))
                .expect("Failed to create empty data")
        })
    }
}

#[cfg(unix)]
fn sync_parent_dir(path: &Path) -> Result<(), StoreError> {
    if let Some(parent) = path.parent() {
        fs::File::open(parent)
            .and_then(|dir| dir.sync_all())
            .map_err(StoreError::Io)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent_dir(_path: &Path) -> Result<(), StoreError> {
    Ok(())
}

/// Errors that can occur during store operations.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Deserialization error: {0}")]
    Deserialization(serde_json::Error),

    #[error("Config file not found: {}", .0.display())]
    NotFound(PathBuf),
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
