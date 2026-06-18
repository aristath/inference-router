//! Static front-end assets embedded into the binary.
//!
//! The dashboard ships htmx + idiomorph for live updates. We vendor them and
//! `include_str!` them straight into the executable so the router has no
//! runtime asset directory and works on fully offline / isolated boxes —
//! consistent with the single-binary deployment model.
//!
//! Versions: htmx 2.0.4, idiomorph 0.7.3 (the `-ext` build bundles the core
//! morph algorithm *and* registers the htmx `morph` swap/OOB extension).

use axum::extract::Path;
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;

const HTMX: &str = include_str!("vendor/htmx.min.js");
const IDIOMORPH_EXT: &str = include_str!("vendor/idiomorph-ext.min.js");

/// Serves a vendored asset by file name (`GET /assets/{file}`).
///
/// Only an explicit allow-list of file names resolves; anything else is a
/// 404. The list is tiny and fixed, so a match is clearer (and safer — no
/// path traversal surface) than a directory lookup.
pub async fn serve_asset(Path(file): Path<String>) -> impl IntoResponse {
    let body = match file.as_str() {
        "htmx.min.js" => HTMX,
        "idiomorph-ext.min.js" => IDIOMORPH_EXT,
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    (
        [
            (header::CONTENT_TYPE, "text/javascript; charset=utf-8"),
            // Vendored, versioned, immutable for the life of the binary.
            (header::CACHE_CONTROL, "public, max-age=31536000, immutable"),
        ],
        body,
    )
        .into_response()
}
