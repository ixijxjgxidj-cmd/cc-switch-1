//! Codex 空回复守卫与检测器
//!
//! 解决模型思考耗时数十秒却返回 0 Token 空回复（`output: []` 且 0 Token 消耗）的问题。
//! 该模块用于检测上游 SSE 流是否在未产生实质输出前就以空回复结束，
//! 供转发层拦截并自动就地重新发起请求。

use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use serde_json::Value;
use crate::proxy::sse::SseBuffer;

pub(crate) enum StreamContentCheck {
    /// 检测到了实质性文本/工具输出，包含已缓冲的首部字节和剩余流
    HasContent {
        buffered: Vec<Bytes>,
        stream: std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
    },
    /// 流结束，确认为 0 Token 空回复（未输出实质文本，直接 completed 且 output: []）
    EmptyReply,
    /// 普通结束、错误或无内容但非典型空回复
    Other {
        buffered: Vec<Bytes>,
    },
}

impl std::fmt::Debug for StreamContentCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HasContent { buffered, .. } => f
                .debug_struct("HasContent")
                .field("buffered_chunks", &buffered.len())
                .finish(),
            Self::EmptyReply => write!(f, "EmptyReply"),
            Self::Other { buffered } => f
                .debug_struct("Other")
                .field("buffered_chunks", &buffered.len())
                .finish(),
        }
    }
}

/// 检查流的前段是否产生了实质内容
///
/// 只有在收到首个有效 content（`response.output_text.delta`、`response.output_item.added` 或工具调用）时才返回 `HasContent`。
/// 若流结束且为 `response.completed` 空回复，则返回 `EmptyReply`，供上层发起自动重试。
pub(crate) async fn check_stream_first_content(
    mut stream: std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>,
) -> StreamContentCheck {
    let mut sse = SseBuffer::new();
    let mut buffered_chunks: Vec<Bytes> = Vec::new();
    let mut is_empty_completed = false;

    while let Some(chunk_res) = stream.next().await {
        let Ok(chunk) = chunk_res else {
            break;
        };
        buffered_chunks.push(chunk.clone());
        sse.push_bytes(&chunk);

        while let Some(block) = sse.next_block() {
            if block.trim().is_empty() {
                continue;
            }

            // 1. 检查是否产生了真正的文本内容增量或工具调用
            if is_productive_content_block(block) {
                return StreamContentCheck::HasContent {
                    buffered: buffered_chunks,
                    stream,
                };
            }

            // 2. 检查是否在没有内容产生的情况下直接收到了 completed
            if block.contains("response.completed") && is_empty_completed_block(block) {
                is_empty_completed = true;
            }
        }
    }

    // 检查 tail
    let tail = sse.take_tail();
    if !tail.trim().is_empty() {
        if is_productive_content_block(&tail) {
            return StreamContentCheck::HasContent {
                buffered: buffered_chunks,
                stream,
            };
        }
        if tail.contains("response.completed") && is_empty_completed_block(&tail) {
            is_empty_completed = true;
        }
    }

    if is_empty_completed {
        StreamContentCheck::EmptyReply
    } else {
        StreamContentCheck::Other {
            buffered: buffered_chunks,
        }
    }
}

/// 判定一个 SSE 事件块是否包含实质性的输出内容（文本 delta、新增输出项或工具调用参数）
pub(crate) fn is_productive_content_block(block: &str) -> bool {
    // 文本增量事件或工具调用增量
    if block.contains("response.output_text.delta")
        || block.contains("response.function_call_arguments.delta")
        || block.contains("response.custom_tool_call_arguments.delta")
        || block.contains("\"delta\":{\"content\":")
    {
        return true;
    }

    // 检查 output_item.added / output_item.done 是否带有非空文本或工具
    for line in block.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            if let Ok(val) = serde_json::from_str::<Value>(data.trim()) {
                let typ = val.get("type").and_then(Value::as_str).unwrap_or("");
                if typ == "response.output_item.added" || typ == "response.output_item.done" {
                    if let Some(item) = val.get("item") {
                        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
                        if item_type == "function_call" || item_type == "custom_tool_call" {
                            return true;
                        }
                        if let Some(content) = item.get("content").and_then(Value::as_array) {
                            for part in content {
                                if let Some(text) = part.get("text").and_then(Value::as_str) {
                                    if !text.is_empty() {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    false
}

/// 判定一个 SSE 事件块是否为 output 为空且使用量为 0 的典型空回复
pub(crate) fn is_empty_completed_block(block: &str) -> bool {
    for line in block.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            if let Ok(val) = serde_json::from_str::<Value>(data.trim()) {
                if val.get("type").and_then(Value::as_str) == Some("response.completed") {
                    let resp = val.get("response").unwrap_or(&val);
                    let empty_output = resp.get("output").map_or(true, |o| {
                        if o.is_null() {
                            return true;
                        }
                        if let Some(arr) = o.as_array() {
                            if arr.is_empty() {
                                return true;
                            }
                            return arr.iter().all(|item| {
                                item.get("content").map_or(true, |c| {
                                    c.is_null()
                                        || (c.is_array() && c.as_array().unwrap().is_empty())
                                })
                            });
                        }
                        false
                    });

                    let zero_tokens = resp.get("usage").map_or(true, |u| {
                        let total = u.get("total_tokens").and_then(Value::as_u64).unwrap_or(0);
                        let output = u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
                        let input = u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
                        output == 0 && (total == 0 || total == input)
                    });

                    return empty_output && zero_tokens;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_detects_empty_completed_reply() {
        let sse_data = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"output\":[]}}\n\n\
                        event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"response\":{\"id\":\"r1\",\"output\":[]}}\n\n\
                        event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"output\":[],\"usage\":{\"total_tokens\":0,\"output_tokens\":0}}}\n\n";

        let stream = Box::pin(futures::stream::once(async move {
            Ok::<Bytes, std::io::Error>(Bytes::from(sse_data))
        }));

        let check = check_stream_first_content(stream).await;
        assert!(matches!(check, StreamContentCheck::EmptyReply));
    }

    #[tokio::test]
    async fn test_detects_productive_text_delta() {
        let sse_data = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"output\":[]}}\n\n\
                        event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello world\"}\n\n";

        let stream = Box::pin(futures::stream::once(async move {
            Ok::<Bytes, std::io::Error>(Bytes::from(sse_data))
        }));

        let check = check_stream_first_content(stream).await;
        match check {
            StreamContentCheck::HasContent { buffered, .. } => {
                let merged = buffered.into_iter().map(|b| String::from_utf8_lossy(&b).to_string()).collect::<String>();
                assert!(merged.contains("response.output_text.delta"));
            }
            _ => panic!("Expected HasContent"),
        }
    }

    #[tokio::test]
    async fn test_detects_empty_reply_with_null_output() {
        let sse_data = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"output\":null,\"usage\":{\"total_tokens\":0,\"output_tokens\":0,\"input_tokens\":0}}}\n\n";

        let stream = Box::pin(futures::stream::once(async move {
            Ok::<Bytes, std::io::Error>(Bytes::from(sse_data))
        }));

        let check = check_stream_first_content(stream).await;
        assert!(matches!(check, StreamContentCheck::EmptyReply));
    }

    #[tokio::test]
    async fn test_detects_function_call_delta() {
        let sse_data = "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"path\\\":\\\"file.rs\\\"}\"}\n\n";

        let stream = Box::pin(futures::stream::once(async move {
            Ok::<Bytes, std::io::Error>(Bytes::from(sse_data))
        }));

        let check = check_stream_first_content(stream).await;
        assert!(matches!(check, StreamContentCheck::HasContent { .. }));
    }
}
