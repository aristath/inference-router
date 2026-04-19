use serde::{de::DeserializeOwned, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::debug;

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
}

impl<T: Clone + DeserializeOwned + Serialize + Send + Sync> JsonStore<T> {
    /// Creates a new store at the given path, loading existing data if present.
    pub fn new(path: PathBuf) -> Self {
        let data = match Self::load_file(&path) {
            Ok(data) => {
                debug!("Loaded config from {}", path.display());
                data
            }
            Err(e) => {
                debug!(
                    "No existing config at {}, will create on first save: {}",
                    path.display(),
                    e,
                );
                Self::empty_data()
            }
        };
        Self {
            path,
            data: std::sync::Mutex::new(data),
        }
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
        let snapshot = self.snapshot();
        let json = serde_json::to_string_pretty(&snapshot).map_err(StoreError::Serialization)?;

        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(StoreError::Io)?;
        }

        // Write to temp file, then rename for atomicity.
        let temp_path = self.path.with_extension("json.tmp");
        {
            let mut file = fs::File::create(&temp_path).map_err(StoreError::Io)?;
            file.write_all(json.as_bytes()).map_err(StoreError::Io)?;
            file.sync_all().map_err(StoreError::Io)?;
        }
        fs::rename(&temp_path, &self.path).map_err(StoreError::Io)?;

        debug!("Saved config to {}", self.path.display());
        Ok(())
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
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::TempDir;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestItem {
        id: String,
        value: i32,
    }

    fn create_test_store(dir: &TempDir) -> JsonStore<Vec<TestItem>> {
        let path = dir.path().join("items.json");
        JsonStore::new(path)
    }

    #[test]
    fn test_store_creates_file_on_save() {
        let dir = tempfile::tempdir().unwrap();
        let store = create_test_store(&dir);

        store.with_mut(|items| items.push(TestItem { id: "a".into(), value: 1 }));
        store.save().unwrap();

        assert!(dir.path().join("items.json").exists());
    }

    #[test]
    fn test_store_loads_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("items.json");

        let items = vec![
            TestItem { id: "a".into(), value: 1 },
            TestItem { id: "b".into(), value: 2 },
        ];
        let json = serde_json::to_string_pretty(&items).unwrap();
        fs::write(&path, json).unwrap();

        let store: JsonStore<Vec<TestItem>> = JsonStore::new(path);
        let loaded = store.snapshot();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "a");
        assert_eq!(loaded[1].value, 2);
    }

    #[test]
    fn test_store_empty_on_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = create_test_store(&dir);

        assert!(store.snapshot().is_empty());
    }

    #[test]
    fn test_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("items.json");
        let store: JsonStore<Vec<TestItem>> = JsonStore::new(path.clone());

        store.with_mut(|items| items.push(TestItem { id: "x".into(), value: 42 }));
        store.save().unwrap();

        let store2: JsonStore<Vec<TestItem>> = JsonStore::new(path);
        let items = store2.snapshot();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, "x");
        assert_eq!(items[0].value, 42);
    }

    #[test]
    fn test_store_atomic_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("items.json");
        let store: JsonStore<Vec<TestItem>> = JsonStore::new(path.clone());

        store.with_mut(|items| items.push(TestItem { id: "initial".into(), value: 0 }));
        store.save().unwrap();

        store.with_mut(|items| items[0].value = 100);
        store.save().unwrap();

        let store2: JsonStore<Vec<TestItem>> = JsonStore::new(path);
        let items = store2.snapshot();
        assert_eq!(items[0].value, 100);

        assert!(!dir.path().join("items.json.tmp").exists());
    }

    #[test]
    fn test_store_preserves_formatting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("items.json");
        let store: JsonStore<Vec<TestItem>> = JsonStore::new(path.clone());

        store.with_mut(|items| items.push(TestItem { id: "fmt".into(), value: 1 }));
        store.save().unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        assert!(contents.contains('\n'));
        assert!(contents.contains(' '));
    }
}
