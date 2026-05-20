use crate::bfloat16::Bf16;
use crate::qwen3_moe::config::Config;
use crate::qwen3_moe::reference_cpu::Qwen35CpuModel;
use axum::extract::State;
use axum::http::{header::AUTHORIZATION, HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::net::TcpListener;

#[derive(Clone)]
pub struct OpenAiServerOptions {
    pub host: String,
    pub port: u16,
    pub model_id: String,
    pub api_key: String,
    pub default_max_tokens: usize,
}

struct ServerState {
    model: Mutex<Qwen35CpuModel>,
    tokenizer: Tokenizer,
    model_id: String,
    api_key: String,
    max_context: usize,
    default_max_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    stream: Option<bool>,
    #[serde(default)]
    prompt_token_ids: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: Value,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

pub fn start_reference_openai_server(
    config: Config,
    weights: HashMap<String, Vec<Bf16>>,
    tokenizer: Tokenizer,
    eos_token_ids: Vec<usize>,
    max_context: usize,
    options: OpenAiServerOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = Qwen35CpuModel::with_eos_token_ids(config, weights, max_context, eos_token_ids)
        .map_err(|err| format!("failed to initialize Qwen3.6 CPU reference model: {}", err))?;
    let state = Arc::new(ServerState {
        model: Mutex::new(model),
        tokenizer,
        model_id: options.model_id.clone(),
        api_key: options.api_key.clone(),
        max_context,
        default_max_tokens: options.default_max_tokens,
    });

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run_server(state, options))
}

async fn run_server(
    state: Arc<ServerState>,
    options: OpenAiServerOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let bind_addr = format!("{}:{}", options.host, options.port);
    let listener = TcpListener::bind(&bind_addr).await?;
    let addr = listener.local_addr()?;
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .with_state(state.clone());

    println!(
        "eLLM Qwen3.6 CPU OpenAI-compatible server listening on http://{}",
        addr
    );
    println!(
        "model_id={}, max_context={}, default_max_tokens={}, api_key={}",
        state.model_id, state.max_context, state.default_max_tokens, state.api_key
    );
    println!("Routes: GET /health, GET /v1/models, POST /v1/chat/completions");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "runtime": "ellm-qwen35-cpu-reference",
        "model": state.model_id,
        "max_context": state.max_context,
    }))
}

async fn models(State(state): State<Arc<ServerState>>, headers: HeaderMap) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_id,
            "object": "model",
            "created": now_unix(),
            "owned_by": "local",
        }],
    }))
    .into_response()
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }

    let model_name = request
        .model
        .clone()
        .unwrap_or_else(|| state.model_id.clone());
    if model_name != state.model_id {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "model_not_found",
            &format!(
                "requested model '{}' is not loaded; available model is '{}'",
                model_name, state.model_id
            ),
        );
    }

    let prompt_tokens = match prompt_tokens(&state.tokenizer, &request) {
        Ok(tokens) => tokens,
        Err(message) => return openai_error(StatusCode::BAD_REQUEST, "bad_request", &message),
    };
    if prompt_tokens.is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "prompt token list must not be empty",
        );
    }
    let prompt_token_count = prompt_tokens.len();
    if prompt_tokens.len() >= state.max_context {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "context_length_exceeded",
            &format!(
                "prompt has {} tokens, but server max context is {}",
                prompt_tokens.len(),
                state.max_context
            ),
        );
    }

    let remaining_context = state.max_context - prompt_tokens.len();
    let max_tokens =
        match requested_max_tokens(&request, state.default_max_tokens, remaining_context) {
            Ok(max_tokens) => max_tokens,
            Err(message) => {
                return openai_error(StatusCode::BAD_REQUEST, "context_length_exceeded", &message)
            }
        };

    let state_for_generation = state.clone();
    let generated = tokio::task::spawn_blocking(move || {
        let mut model = state_for_generation
            .model
            .lock()
            .expect("Qwen3.6 CPU model mutex poisoned");
        model.generate_greedy(&prompt_tokens, max_tokens)
    })
    .await;
    let generated = match generated {
        Ok(tokens) => tokens,
        Err(err) => {
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &format!("generation task failed: {}", err),
            )
        }
    };

    let generated_ids = match to_u32_ids(&generated) {
        Ok(ids) => ids,
        Err(message) => {
            return openai_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", &message)
        }
    };
    let content = match state.tokenizer.decode(&generated_ids, true) {
        Ok(text) => text,
        Err(err) => {
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &format!("tokenizer decode failed: {}", err),
            )
        }
    };
    let finish_reason = finish_reason(&state, &generated, max_tokens);
    let usage = Usage {
        prompt_tokens: prompt_token_count,
        completion_tokens: generated.len(),
        total_tokens: prompt_token_count + generated.len(),
    };

    if request.stream.unwrap_or(false) {
        stream_response(&model_name, content, finish_reason, usage)
    } else {
        completion_response(&model_name, content, finish_reason, usage)
    }
}

fn completion_response(
    model_name: &str,
    content: String,
    finish_reason: &'static str,
    usage: Usage,
) -> Response {
    Json(json!({
        "id": format!("chatcmpl-{}", now_unix_nanos()),
        "object": "chat.completion",
        "created": now_unix(),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }))
    .into_response()
}

fn stream_response(
    model_name: &str,
    content: String,
    finish_reason: &'static str,
    usage: Usage,
) -> Response {
    let completion_id = format!("chatcmpl-{}", now_unix_nanos());
    let created = now_unix();
    let content_chunk = json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": null,
        }],
    });
    let final_chunk = json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    });
    let events: Vec<Result<Event, Infallible>> = vec![
        Ok(Event::default().data(content_chunk.to_string())),
        Ok(Event::default().data(final_chunk.to_string())),
        Ok(Event::default().data("[DONE]")),
    ];
    let stream = stream::iter(events);
    Sse::new(stream).into_response()
}

fn authorize(state: &ServerState, headers: &HeaderMap) -> Result<(), Response> {
    if state.api_key.is_empty() {
        return Ok(());
    }
    let expected = format!("Bearer {}", state.api_key);
    let authorized = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .map(|value| value == expected)
        .unwrap_or(false);
    if authorized {
        Ok(())
    } else {
        Err(openai_error(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing or invalid Authorization header",
        ))
    }
}

fn prompt_tokens(
    tokenizer: &Tokenizer,
    request: &ChatCompletionRequest,
) -> Result<Vec<usize>, String> {
    if let Some(tokens) = request.prompt_token_ids.as_ref() {
        return Ok(tokens.clone());
    }
    let messages = request
        .messages
        .as_ref()
        .filter(|messages| !messages.is_empty())
        .ok_or_else(|| "messages must be a non-empty list".to_string())?;
    let prompt = render_chat_prompt(messages);
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|err| format!("tokenizer encode failed: {}", err))?;
    Ok(encoding
        .get_ids()
        .iter()
        .map(|&token| token as usize)
        .collect())
}

fn render_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let role = match message.role.as_str() {
            "system" | "user" | "assistant" | "tool" => message.role.as_str(),
            _ => "user",
        };
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&content_as_text(&message.content));
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn content_as_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.get("content").and_then(Value::as_str))
            })
            .collect::<Vec<_>>()
            .join(""),
        other => other.to_string(),
    }
}

fn requested_max_tokens(
    request: &ChatCompletionRequest,
    default_max_tokens: usize,
    remaining_context: usize,
) -> Result<usize, String> {
    let requested = request
        .max_tokens
        .or(request.max_completion_tokens)
        .unwrap_or(default_max_tokens.min(remaining_context));
    if requested == 0 {
        return Err("max_tokens must be greater than zero".to_string());
    }
    if requested > remaining_context {
        return Err(format!(
            "requested max_tokens={}, but only {} tokens remain in the context window",
            requested, remaining_context
        ));
    }
    Ok(requested)
}

fn finish_reason(
    state: &ServerState,
    generated_tokens: &[usize],
    max_tokens: usize,
) -> &'static str {
    let last_is_eos = generated_tokens
        .last()
        .map(|&token| {
            state
                .model
                .lock()
                .expect("Qwen3.6 CPU model mutex poisoned")
                .is_eos_token(token)
        })
        .unwrap_or(false);
    if last_is_eos {
        "stop"
    } else if generated_tokens.len() >= max_tokens {
        "length"
    } else {
        "stop"
    }
}

fn to_u32_ids(tokens: &[usize]) -> Result<Vec<u32>, String> {
    tokens
        .iter()
        .map(|&token| u32::try_from(token).map_err(|_| format!("token id {} exceeds u32", token)))
        .collect()
}

fn openai_error(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
            }
        })),
    )
        .into_response()
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn now_unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}
