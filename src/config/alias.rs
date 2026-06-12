use serde::{Deserialize, Serialize};

/// A user-defined alias that maps an arbitrary model name to one of the
/// configured models. Clients can request inference using `alias`; the router
/// resolves it to `target` before loading/serving. Aliases are also exposed on
/// the `/v1/models` endpoint according to the configured exposure mode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelAlias {
    /// The alias name clients use (must be unique and must not collide with a
    /// real model id).
    pub alias: String,
    /// The id of the configured model this alias resolves to.
    pub target: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_serde() {
        let a = ModelAlias {
            alias: "gpt-4o".into(),
            target: "qwen3-32b".into(),
        };
        let s = serde_json::to_string(&a).unwrap();
        let back: ModelAlias = serde_json::from_str(&s).unwrap();
        assert_eq!(a, back);
    }
}
