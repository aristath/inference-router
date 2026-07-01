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

    store.with_mut(|items| {
        items.push(TestItem {
            id: "a".into(),
            value: 1,
        })
    });
    store.save().unwrap();

    assert!(dir.path().join("items.json").exists());
}

#[test]
fn test_store_loads_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("items.json");

    let items = vec![
        TestItem {
            id: "a".into(),
            value: 1,
        },
        TestItem {
            id: "b".into(),
            value: 2,
        },
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

    store.with_mut(|items| {
        items.push(TestItem {
            id: "x".into(),
            value: 42,
        })
    });
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

    store.with_mut(|items| {
        items.push(TestItem {
            id: "initial".into(),
            value: 0,
        })
    });
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

    store.with_mut(|items| {
        items.push(TestItem {
            id: "fmt".into(),
            value: 1,
        })
    });
    store.save().unwrap();

    let contents = fs::read_to_string(&path).unwrap();
    assert!(contents.contains('\n'));
    assert!(contents.contains(' '));
}
