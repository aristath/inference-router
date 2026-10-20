//! Read routing metadata without retaining queued prompt bodies in RAM. Small
//! requests stay in memory; large requests arriving during media spill to an
//! unnamed private file beside the router configuration and unlink on drop.
use axum::body::{Body, Bytes};
use futures_util::StreamExt;
use serde::Deserialize;
use std::io::{BufReader, Seek};
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub enum RoutingBody {
    Memory(Bytes),
    Disk(std::fs::File, Option<String>),
}

impl RoutingBody {
    pub async fn read(body: Body, max: usize, spool: Option<&Path>) -> Result<Self, String> {
        let Some(directory) = spool else {
            return axum::body::to_bytes(body, max)
                .await
                .map(Self::Memory)
                .map_err(|e| e.to_string());
        };
        let mut stream = body.into_data_stream();
        let mut memory = Vec::new();
        let mut file = None;
        let mut size = 0usize;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| e.to_string())?;
            size = size
                .checked_add(chunk.len())
                .ok_or("request is too large")?;
            if size > max {
                return Err("request exceeds the configured body limit".into());
            }
            if file.is_none() && size > 64 * 1024 {
                tokio::fs::create_dir_all(directory)
                    .await
                    .map_err(|e| e.to_string())?;
                let dir = directory.to_owned();
                let temp = tokio::task::spawn_blocking(move || tempfile::tempfile_in(dir))
                    .await
                    .map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
                let mut writer = tokio::fs::File::from_std(temp);
                writer.write_all(&memory).await.map_err(|e| e.to_string())?;
                memory.clear();
                memory.shrink_to_fit();
                file = Some(writer);
            }
            if let Some(writer) = &mut file {
                writer.write_all(&chunk).await.map_err(|e| e.to_string())?;
            } else {
                memory.extend_from_slice(&chunk);
            }
        }
        if let Some(file) = file {
            let mut file = file.into_std().await;
            tokio::task::spawn_blocking(move || {
                file.rewind().map_err(|e| e.to_string())?;
                #[derive(Deserialize)]
                struct Metadata {
                    model: String,
                }
                let model = serde_json::from_reader::<_, Metadata>(BufReader::new(&mut file))
                    .ok()
                    .map(|v| v.model);
                file.rewind().map_err(|e| e.to_string())?;
                Ok(Self::Disk(file, model))
            })
            .await
            .map_err(|e| e.to_string())?
        } else {
            Ok(Self::Memory(memory.into()))
        }
    }

    pub fn model(&self) -> Option<String> {
        match self {
            Self::Memory(bytes) => super::body_peek::extract_model(bytes),
            Self::Disk(_, model) => model.clone(),
        }
    }

    pub async fn into_bytes(self) -> Result<Bytes, std::io::Error> {
        match self {
            Self::Memory(bytes) => Ok(bytes),
            Self::Disk(file, _) => {
                let mut bytes = Vec::new();
                tokio::fs::File::from_std(file)
                    .read_to_end(&mut bytes)
                    .await?;
                Ok(bytes.into())
            }
        }
    }
}
