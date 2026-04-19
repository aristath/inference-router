use serde::{de::DeserializeOwned, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::debug;

/// A generic file-backed JSON store.
///
/// Provides atomic load/save operations using write-to-temp-then-rename.
/// Thread-safe via `std::sync::Mutex`.
pub struct JsonStore<T: DeserializeOwned + Serialize + Send + Sync> {
    path: PathBuf,
    data: std::sync::Mutex<T>,
}

impl<T: DeserializeOwned + Serialize + Send + Sync> JsonStore<T> {
    /// Creates a new store at the given path, loading existing data if present.
    pub fn new(path: PathBuf) -> Self {
        let data = match Self::load_file(&path) {
            Ok(data) => {
                debug!("Loaded config from {}", path.display());
                data
            }
            Err(e) => {
                debug!("No existing config at {}, will create on first save: {}", path.display(), e);
                Self::empty_data()
            }
        };
        Self {
            path,
            data: std::sync::Mutex::new(data),
        }
    }

    /// Creates a store with empty initial data (for testing).
    pub fn with_data(path: PathBuf, data: T) -> Self {
        Self {
            path,
            data: std::sync::Mutex::new(data),
        }
    }

    /// Returns the path this store writes to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reads the current data under the lock.
    pub fn read(&self) -> std::sync::MutexGuard<'_, T> {
        self.data.lock().expect("JsonStore lock poisoned")
    }

    /// Mutates the data under the lock.
    pub fn write(&self) -> std::sync::MutexGuard<'_, T> {
        self.data.lock().expect("JsonStore lock poisoned")
    }

    /// Saves the current data to disk atomically.
    pub fn save(&self) -> Result<(), StoreError> {
        let data = self.read();
        let json = serde_json::to_string_pretty(&*data)
            .map_err(|e| StoreError::Serialization(e))?;

        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| StoreError::Io(e))?;
        }

        // Write to temp file, then rename for atomicity
        let temp_path = self.path.with_extension("json.tmp");
        {
            let mut file = fs::File::create(&temp_path)
                .map_err(|e| StoreError::Io(e))?;
            file.write_all(json.as_bytes())
                .map_err(|e| StoreError::Io(e))?;
            file.sync_all()
                .map_err(|e| StoreError::Io(e))?;
        }
        fs::rename(&temp_path, &self.path)
            .map_err(|e| StoreError::Io(e))?;

        debug!("Saved config to {}", self.path.display());
        Ok(())
    }

    /// Loads a single file from disk.
    fn load_file(path: &Path) -> Result<T, StoreError> {
        if !path.exists() {
            return Err(StoreError::NotFound(path.to_path_buf()));
        }
        let contents = fs::read_to_string(path)
            .map_err(|e| StoreError::Io(e))?;
        let data = serde_json::from_str(&contents)
            .map_err(|e| StoreError::Deserialization(e))?;
        Ok(data)
    }

    /// Returns the default empty data.
    fn empty_data() -> T {
        // This requires T to implement Default, which we can't enforce.
        // Instead, we use a workaround: try to load an empty JSON value.
        // For Vec<T>, this means []. For other types, we'll need specific handling.
        // Since most of our stores use Vec<T>, this works for those.
        serde_json::from_value(serde_json::Value::Array(vec![]))
            .unwrap_or_else(|_| {
                // Fallback: try object
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

        let mut items = store.write();
        items.push(TestItem { id: "a".into(), value: 1 });
        drop(items);

        store.save().unwrap();

        assert!(dir.path().join("items.json").exists());
    }

    #[test]
    fn test_store_loads_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("items.json");

        // Create file manually
        let items = vec![
            TestItem { id: "a".into(), value: 1 },
            TestItem { id: "b".into(), value: 2 },
        ];
        let json = serde_json::to_string_pretty(&items).unwrap();
        fs::write(&path, json).unwrap();

        // Store should load it
        let store: JsonStore<Vec<TestItem>> = JsonStore::new(path);
        let loaded = store.read();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "a");
        assert_eq!(loaded[1].value, 2);
    }

    #[test]
    fn test_store_empty_on_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = create_test_store(&dir);

        let items = store.read();
        assert!(items.is_empty());
    }

    #[test]
    fn test_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = create_test_store(&dir);

        // Write
        {
            let mut items = store.write();
            items.push(TestItem { id: "x".into(), value: 42 });
        }
        store.save().unwrap();

        // Reload from disk
        let store2: JsonStore<Vec<TestItem>> = JsonStore::new(store.path().to_path_buf());
        let items = store2.read();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, "x");
        assert_eq!(items[0].value, 42);
    }

    #[test]
    fn test_store_atomic_write() {
        let dir = tempfile::tempdir().unwrap();
        let store = create_test_store(&dir);

        // Save initial data
        {
            let mut items = store.write();
            items.push(TestItem { id: "initial".into(), value: 0 });
        }
        store.save().unwrap();

        // Modify and save again
        {
            let mut items = store.write();
            items[0].value = 100;
        }
        store.save().unwrap();

        // Verify the final state is correct
        let store2: JsonStore<Vec<TestItem>> = JsonStore::new(store.path().to_path_buf());
        let items = store2.read();
        assert_eq!(items[0].value, 100);

        // No temp file should be left behind
        assert!(!dir.path().join("items.json.tmp").exists());
    }

    #[test]
    fn test_store_preserves_formatting() {
        let dir = tempfile::tempdir().unwrap();
        let store = create_test_store(&dir);

        {
            let mut items = store.write();
            items.push(TestItem { id: "fmt".into(), value: 1 });
        }
        store.save().unwrap();

        let contents = fs::read_to_string(store.path()).unwrap();
        // Should be pretty-printed (indented)
        assert!(contents.contains('\n'));
        assert!(contents.contains(' '));
    }
}
