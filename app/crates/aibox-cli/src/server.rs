//! SERV-04 : Serveur HTTP compatible OpenAI (`/v1/*`) et Ollama (`/api/*`) sur le port 11434.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use aibox_kernels::{select_backend, TernaryHadamardBackend};

use crate::engine::{
    build_synthetic_tensor, EmbeddingTable, LmHead, ModelConfig, TernaryLayerWeights,
    TransformerModel,
};
use crate::json::{escape_json_string, parse_json, JsonValue};
use crate::models;
use crate::sampling::SamplingParams;
use crate::tokenizer::{SimpleTokenizer, TokenStreamDecoder};

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 11434;
const DEFAULT_MODEL_NAME: &str = "aibox-ternary-demo";
const MAX_GENERATION_TOKENS: usize = 4096;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_dir: Option<PathBuf>,
    pub use_gpu: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
            model_dir: None,
            use_gpu: false,
        }
    }
}

struct ServerState {
    backend: Box<dyn TernaryHadamardBackend>,
    model: TransformerModel,
    tokenizer: Option<SimpleTokenizer>,
}

struct HttpRequest {
    method: String,
    path: String,
    body: String,
}

struct CompletionRequest {
    model: String,
    prompt: String,
    stream: bool,
    max_tokens: usize,
    sampling: SamplingParams,
}

struct InferenceOutput {
    model: String,
    full_text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    total_duration_ns: u128,
}

pub fn bind_server(config: &ServerConfig) -> Result<TcpListener, String> {
    let addr = format!("{}:{}", config.host, config.port);
    TcpListener::bind(&addr).map_err(|err| format!("Impossible d'écouter sur {addr}: {err}"))
}

pub fn run_server(config: ServerConfig) -> Result<(), String> {
    let listener = bind_server(&config)?;
    let local_addr = listener
        .local_addr()
        .map_err(|err| format!("Erreur lecture adresse locale: {err}"))?;
    println!("Serveur AI Box (OpenAI + Ollama) à l'écoute sur http://{local_addr}");
    println!("Endpoints : /v1/chat/completions, /v1/models, /api/chat, /api/generate, /api/tags");
    serve_listener_with_config(listener, &config)
}

#[cfg(test)]
pub fn serve_listener(listener: TcpListener) -> Result<(), String> {
    serve_listener_with_config(listener, &ServerConfig::default())
}

pub fn serve_listener_with_config(listener: TcpListener, config: &ServerConfig) -> Result<(), String> {
    let state = Arc::new(build_server_state(config)?);
    for incoming in listener.incoming() {
        let stream = match incoming {
            Ok(stream) => stream,
            Err(err) => {
                eprintln!("Erreur connexion TCP : {err}");
                continue;
            }
        };
        let state_ref = Arc::clone(&state);
        thread::spawn(move || {
            if let Err(err) = handle_connection(stream, &state_ref) {
                eprintln!("Erreur traitement requête HTTP : {err}");
            }
        });
    }
    Ok(())
}

fn build_server_state(server_cfg: &ServerConfig) -> Result<ServerState, String> {
    if let Some(ref model_dir) = server_cfg.model_dir {
        println!("Chargement du modèle réel depuis {}...", model_dir.display());
        let model = TransformerModel::load_from_dir(model_dir)?;
        let tokenizer = SimpleTokenizer::from_file(&model_dir.join("tokenizer.json"))?;
        return Ok(ServerState {
            backend: select_backend(server_cfg.use_gpu),
            model,
            tokenizer: Some(tokenizer),
        });
    }

    let config = ModelConfig {
        hidden_size: 256,
        intermediate_size: 512,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 64,
        vocab_size: 256,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        use_hadamard: true,
        ..ModelConfig::default()
    };
    let kv_dim = config.num_key_value_heads * config.head_dim();
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for idx in 0..config.num_hidden_layers {
        layers.push(TernaryLayerWeights {
            attn_norm: vec![1.0; config.hidden_size],
            block: crate::engine::LayerBlock::Attention(crate::engine::AttentionBlockWeights {
                attn_q: build_synthetic_tensor(config.hidden_size, config.hidden_size, idx + 1)?,
                attn_k: build_synthetic_tensor(kv_dim, config.hidden_size, idx + 2)?,
                attn_v: build_synthetic_tensor(kv_dim, config.hidden_size, idx + 3)?,
                attn_o: build_synthetic_tensor(config.hidden_size, config.hidden_size, idx + 4)?,
                attn_q_norm: None,
                attn_k_norm: None,
                has_q_gate: false,
            }),
            ffn_norm: vec![1.0; config.hidden_size],
            ffn_gate: build_synthetic_tensor(config.intermediate_size, config.hidden_size, idx + 5)?,
            ffn_up: build_synthetic_tensor(config.intermediate_size, config.hidden_size, idx + 6)?,
            ffn_down: build_synthetic_tensor(config.hidden_size, config.intermediate_size, idx + 7)?,
        });
    }
    let demo_model = TransformerModel {
        token_embeddings: EmbeddingTable::F32(vec![0.02; config.vocab_size * config.hidden_size]),
        layers,
        final_norm: vec![1.0; config.hidden_size],
        lm_head: LmHead::Ternary(build_synthetic_tensor(config.vocab_size, config.hidden_size, 99)?),
        config,
    };

    Ok(ServerState {
        backend: select_backend(server_cfg.use_gpu),
        model: demo_model,
        tokenizer: None,
    })
}

fn handle_connection(mut stream: TcpStream, state: &ServerState) -> Result<(), String> {
    let req = read_http_request(&mut stream)?;
    let path = req.path.split('?').next().unwrap_or("/");

    if req.method == "OPTIONS" {
        return write_http_response(&mut stream, "204 No Content", "text/plain", "");
    }

    match (req.method.as_str(), path) {
        ("GET", "/") | ("GET", "/health") => {
            write_http_response(&mut stream, "200 OK", "text/plain; charset=utf-8", "aibox is running")
        }
        ("GET", "/v1/models") => handle_openai_models(&mut stream),
        ("GET", "/api/tags") => handle_ollama_tags(&mut stream),
        ("POST", "/v1/chat/completions") => handle_openai_chat(&mut stream, &req.body, state),
        ("POST", "/api/chat") => handle_ollama_chat(&mut stream, &req.body, state),
        ("POST", "/api/generate") => handle_ollama_generate(&mut stream, &req.body, state),
        _ => write_json_error(&mut stream, "404 Not Found", &format!("Route inconnue: {} {}", req.method, path)),
    }
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut request_line = String::new();
    let bytes = reader.read_line(&mut request_line).map_err(|e| e.to_string())?;
    if bytes == 0 {
        return Err("Connexion fermée avant la ligne de requête".to_string());
    }

    let mut parts = request_line.trim_end().split_whitespace();
    let method = parts.next().unwrap_or("GET").to_uppercase();
    let path = parts.next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            if name.trim().eq_ignore_ascii_case("Content-Length") {
                content_length = value.trim().parse::<usize>().unwrap_or(0);
            }
        }
    }

    let mut body_buf = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body_buf).map_err(|e| e.to_string())?;
    }
    let body = String::from_utf8_lossy(&body_buf).to_string();

    Ok(HttpRequest { method, path, body })
}

fn list_available_models() -> Vec<String> {
    let mut names: Vec<String> = models::list_cached_models()
        .unwrap_or_default()
        .into_iter()
        .filter(|(_, complete)| *complete)
        .map(|(name, _)| name)
        .collect();
    if names.is_empty() {
        names.push(DEFAULT_MODEL_NAME.to_string());
    }
    names
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn handle_openai_models(stream: &mut TcpStream) -> Result<(), String> {
    let created = unix_timestamp_secs();
    let items: Vec<String> = list_available_models()
        .into_iter()
        .map(|id| {
            format!(
                "{{\"id\":{},\"object\":\"model\",\"created\":{created},\"owned_by\":\"aibox\"}}",
                escape_json_string(&id)
            )
        })
        .collect();
    let body = format!("{{\"object\":\"list\",\"data\":[{}]}}", items.join(","));
    write_http_response(stream, "200 OK", "application/json", &body)
}

fn handle_ollama_tags(stream: &mut TcpStream) -> Result<(), String> {
    let items: Vec<String> = list_available_models()
        .into_iter()
        .map(|name| {
            let escaped = escape_json_string(&name);
            format!(
                "{{\"name\":{escaped},\"model\":{escaped},\"modified_at\":\"2026-01-01T00:00:00Z\",\"size\":0}}"
            )
        })
        .collect();
    let body = format!("{{\"models\":[{}]}}", items.join(","));
    write_http_response(stream, "200 OK", "application/json", &body)
}

fn handle_openai_chat(stream: &mut TcpStream, body: &str, state: &ServerState) -> Result<(), String> {
    let req = match parse_completion_request(body, false, true) {
        Ok(req) => req,
        Err(err) => return write_json_error(stream, "400 Bad Request", &err),
    };
    if req.stream {
        return stream_openai_sse(stream, &req, state);
    }

    let output = run_inference(&req, state)?;
    let created = unix_timestamp_secs();
    let total_tokens = output.prompt_tokens + output.completion_tokens;
    let payload = format!(
        "{{\"id\":\"chatcmpl-aibox-{created}\",\"object\":\"chat.completion\",\"created\":{created},\
         \"model\":{},\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}}},\
         \"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}}}",
        escape_json_string(&output.model),
        escape_json_string(&output.full_text),
        output.prompt_tokens,
        output.completion_tokens,
        total_tokens
    );
    write_http_response(stream, "200 OK", "application/json", &payload)
}

fn handle_ollama_chat(stream: &mut TcpStream, body: &str, state: &ServerState) -> Result<(), String> {
    let req = match parse_completion_request(body, true, true) {
        Ok(req) => req,
        Err(err) => return write_json_error(stream, "400 Bad Request", &err),
    };
    if req.stream {
        return stream_ollama_ndjson(stream, &req, state, true);
    }

    let output = run_inference(&req, state)?;
    let payload = format!(
        "{{\"model\":{},\"created_at\":\"2026-01-01T00:00:00Z\",\
         \"message\":{{\"role\":\"assistant\",\"content\":{}}},\
         \"done\":true,\"done_reason\":\"stop\",\"total_duration\":{},\
         \"prompt_eval_count\":{},\"eval_count\":{}}}",
        escape_json_string(&output.model),
        escape_json_string(&output.full_text),
        output.total_duration_ns,
        output.prompt_tokens,
        output.completion_tokens
    );
    write_http_response(stream, "200 OK", "application/json", &payload)
}

fn handle_ollama_generate(stream: &mut TcpStream, body: &str, state: &ServerState) -> Result<(), String> {
    let req = match parse_completion_request(body, true, false) {
        Ok(req) => req,
        Err(err) => return write_json_error(stream, "400 Bad Request", &err),
    };
    if req.stream {
        return stream_ollama_ndjson(stream, &req, state, false);
    }

    let output = run_inference(&req, state)?;
    let payload = format!(
        "{{\"model\":{},\"created_at\":\"2026-01-01T00:00:00Z\",\
         \"response\":{},\"done\":true,\"done_reason\":\"stop\",\"total_duration\":{},\
         \"prompt_eval_count\":{},\"eval_count\":{}}}",
        escape_json_string(&output.model),
        escape_json_string(&output.full_text),
        output.total_duration_ns,
        output.prompt_tokens,
        output.completion_tokens
    );
    write_http_response(stream, "200 OK", "application/json", &payload)
}

fn parse_completion_request(
    body: &str,
    default_stream: bool,
    is_chat: bool,
) -> Result<CompletionRequest, String> {
    let root = parse_json(body)?;
    let obj = root
        .as_object()
        .ok_or_else(|| "Le corps JSON doit être un objet".to_string())?;

    let model = obj
        .get("model")
        .and_then(JsonValue::as_str)
        .unwrap_or(DEFAULT_MODEL_NAME)
        .to_string();
    let stream = obj
        .get("stream")
        .and_then(JsonValue::as_bool)
        .unwrap_or(default_stream);

    let prompt = if is_chat {
        extract_chat_prompt(obj)?
    } else {
        obj.get("prompt")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| "Champ 'prompt' manquant".to_string())?
            .to_string()
    };

    let options_obj = obj.get("options").and_then(JsonValue::as_object);
    let max_tokens = obj
        .get("max_tokens")
        .or_else(|| options_obj.and_then(|opts| opts.get("num_predict")))
        .and_then(JsonValue::as_u64)
        .map(|v| (v as usize).clamp(1, MAX_GENERATION_TOKENS))
        .unwrap_or(16);

    let sampling = parse_sampling_params(obj, options_obj);
    Ok(CompletionRequest {
        model,
        prompt,
        stream,
        max_tokens,
        sampling,
    })
}

fn extract_chat_prompt(obj: &BTreeMap<String, JsonValue>) -> Result<String, String> {
    let messages = obj
        .get("messages")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| "Champ 'messages' manquant ou invalide".to_string())?;
    if messages.is_empty() {
        return Err("La liste 'messages' ne peut pas être vide".to_string());
    }

    let mut lines = Vec::with_capacity(messages.len());
    for msg in messages {
        let msg_obj = msg
            .as_object()
            .ok_or_else(|| "Chaque message doit être un objet JSON".to_string())?;
        let role = msg_obj.get("role").and_then(JsonValue::as_str).unwrap_or("user");
        let content = msg_obj.get("content").and_then(JsonValue::as_str).unwrap_or("");
        lines.push(format!("{role}: {content}"));
    }
    Ok(lines.join("\n"))
}

fn parse_sampling_params(
    obj: &BTreeMap<String, JsonValue>,
    options: Option<&BTreeMap<String, JsonValue>>,
) -> SamplingParams {
    let mut params = SamplingParams::default();
    if let Some(temp) = obj
        .get("temperature")
        .or_else(|| options.and_then(|o| o.get("temperature")))
        .and_then(JsonValue::as_f64)
    {
        params.temperature = temp as f32;
    }
    if let Some(top_p) = obj
        .get("top_p")
        .or_else(|| options.and_then(|o| o.get("top_p")))
        .and_then(JsonValue::as_f64)
    {
        params.top_p = top_p as f32;
    }
    if let Some(top_k) = options.and_then(|o| o.get("top_k")).and_then(JsonValue::as_u64) {
        params.top_k = top_k as usize;
    }
    params
}

fn encode_prompt_ids(prompt: &str, state: &ServerState) -> Vec<usize> {
    let vocab_size = state.model.config.vocab_size;
    if let Some(ref tok) = state.tokenizer {
        let ids = tok.encode_chat(prompt);
        if ids.is_empty() { vec![1] } else { ids }
    } else if prompt.is_empty() {
        vec![1]
    } else {
        prompt
            .bytes()
            .take(64)
            .map(|byte| (byte as usize) % vocab_size)
            .collect()
    }
}

fn run_inference(req: &CompletionRequest, state: &ServerState) -> Result<InferenceOutput, String> {
    let prompt_ids = encode_prompt_ids(&req.prompt, state);
    let eos_id = state.tokenizer.as_ref().and_then(SimpleTokenizer::eos_token_id);
    let stats = state.model.generate_tokens(
        &prompt_ids,
        req.max_tokens,
        eos_id,
        req.sampling,
        &*state.backend,
    )?;

    let full_text = if let Some(ref tok) = state.tokenizer {
        tok.decode(&stats.tokens)
    } else {
        stats
            .tokens
            .iter()
            .enumerate()
            .map(|(idx, tok)| {
                if idx == 0 {
                    format!("[tmac:{tok}]")
                } else {
                    format!(" tok_{tok}")
                }
            })
            .collect::<Vec<_>>()
            .concat()
    };
    let total_duration_ns = ((stats.prefill_ms + stats.decode_ms) * 1_000_000.0) as u128;

    Ok(InferenceOutput {
        model: req.model.clone(),
        full_text,
        prompt_tokens: prompt_ids.len(),
        completion_tokens: stats.tokens.len(),
        total_duration_ns,
    })
}

fn stream_openai_sse(
    stream: &mut TcpStream,
    req: &CompletionRequest,
    state: &ServerState,
) -> Result<(), String> {
    let headers = concat!(
        "HTTP/1.1 200 OK\r\n",
        "Content-Type: text/event-stream; charset=utf-8\r\n",
        "Cache-Control: no-cache\r\n",
        "Connection: close\r\n",
        "Access-Control-Allow-Origin: *\r\n",
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n",
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n"
    );
    stream.write_all(headers.as_bytes()).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;

    let created = unix_timestamp_secs();
    let model_json = escape_json_string(&req.model);
    let prompt_ids = encode_prompt_ids(&req.prompt, state);
    let eos_id = state.tokenizer.as_ref().and_then(SimpleTokenizer::eos_token_id);
    let mut decoder = state.tokenizer.as_ref().map(TokenStreamDecoder::new);
    let mut tok_idx = 0usize;

    let write_sse_delta = |s: &mut TcpStream, text: &str| -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }
        let line = format!(
            "data: {{\"id\":\"chatcmpl-aibox-{created}\",\"object\":\"chat.completion.chunk\",\
             \"created\":{created},\"model\":{model_json},\
             \"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}},\"finish_reason\":null}}]}}\n\n",
            escape_json_string(text)
        );
        s.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        s.flush().map_err(|e| e.to_string())
    };

    state.model.generate_tokens_streaming(
        &prompt_ids,
        req.max_tokens,
        eos_id,
        req.sampling,
        &*state.backend,
        |tok| {
            let chunk = if let Some(ref mut dec) = decoder {
                dec.push(tok)
            } else if tok_idx == 0 {
                format!("[tmac:{tok}]")
            } else {
                format!(" tok_{tok}")
            };
            tok_idx += 1;
            write_sse_delta(stream, &chunk)
        },
    )?;

    if let Some(ref mut dec) = decoder {
        write_sse_delta(stream, &dec.finish())?;
    }

    let final_chunk = format!(
        "data: {{\"id\":\"chatcmpl-aibox-{created}\",\"object\":\"chat.completion.chunk\",\
         \"created\":{created},\"model\":{model_json},\
         \"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n\
         data: [DONE]\n\n"
    );
    stream.write_all(final_chunk.as_bytes()).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn stream_ollama_ndjson(
    stream: &mut TcpStream,
    req: &CompletionRequest,
    state: &ServerState,
    is_chat: bool,
) -> Result<(), String> {
    let headers = concat!(
        "HTTP/1.1 200 OK\r\n",
        "Content-Type: application/x-ndjson; charset=utf-8\r\n",
        "Connection: close\r\n",
        "Access-Control-Allow-Origin: *\r\n",
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n",
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n"
    );
    stream.write_all(headers.as_bytes()).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;

    let model_json = escape_json_string(&req.model);
    let prompt_ids = encode_prompt_ids(&req.prompt, state);
    let eos_id = state.tokenizer.as_ref().and_then(SimpleTokenizer::eos_token_id);
    let mut decoder = state.tokenizer.as_ref().map(TokenStreamDecoder::new);
    let mut tok_idx = 0usize;

    let write_ndjson_chunk = |s: &mut TcpStream, text: &str| -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }
        let escaped = escape_json_string(text);
        let line = if is_chat {
            format!(
                "{{\"model\":{model_json},\"created_at\":\"2026-01-01T00:00:00Z\",\
                 \"message\":{{\"role\":\"assistant\",\"content\":{escaped}}},\"done\":false}}\n"
            )
        } else {
            format!(
                "{{\"model\":{model_json},\"created_at\":\"2026-01-01T00:00:00Z\",\
                 \"response\":{escaped},\"done\":false}}\n"
            )
        };
        s.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        s.flush().map_err(|e| e.to_string())
    };

    let stats = state.model.generate_tokens_streaming(
        &prompt_ids,
        req.max_tokens,
        eos_id,
        req.sampling,
        &*state.backend,
        |tok| {
            let chunk = if let Some(ref mut dec) = decoder {
                dec.push(tok)
            } else if tok_idx == 0 {
                format!("[tmac:{tok}]")
            } else {
                format!(" tok_{tok}")
            };
            tok_idx += 1;
            write_ndjson_chunk(stream, &chunk)
        },
    )?;

    if let Some(ref mut dec) = decoder {
        write_ndjson_chunk(stream, &dec.finish())?;
    }

    let total_duration_ns = ((stats.prefill_ms + stats.decode_ms) * 1_000_000.0) as u128;
    let done_line = if is_chat {
        format!(
            "{{\"model\":{model_json},\"created_at\":\"2026-01-01T00:00:00Z\",\
             \"message\":{{\"role\":\"assistant\",\"content\":\"\"}},\"done\":true,\"done_reason\":\"stop\",\
             \"total_duration\":{},\"prompt_eval_count\":{},\"eval_count\":{}}}\n",
            total_duration_ns, prompt_ids.len(), stats.tokens.len()
        )
    } else {
        format!(
            "{{\"model\":{model_json},\"created_at\":\"2026-01-01T00:00:00Z\",\
             \"response\":\"\",\"done\":true,\"done_reason\":\"stop\",\
             \"total_duration\":{},\"prompt_eval_count\":{},\"eval_count\":{}}}\n",
            total_duration_ns, prompt_ids.len(), stats.tokens.len()
        )
    };
    stream.write_all(done_line.as_bytes()).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn write_json_error(stream: &mut TcpStream, status: &str, message: &str) -> Result<(), String> {
    let body = format!("{{\"error\":{}}}", escape_json_string(message));
    write_http_response(stream, status, "application/json", &body)
}

fn write_http_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
) -> Result<(), String> {
    let response = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n\
         {body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    fn spawn_test_server() -> SocketAddr {
        let listener = bind_server(&ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            model_dir: None,
            use_gpu: false,
        })
        .expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        thread::spawn(move || {
            let _ = serve_listener(listener);
        });
        addr
    }

    fn send_raw_http(addr: SocketAddr, method: &str, path: &str, body: &str) -> String {
        let mut stream = TcpStream::connect(addr).expect("connect to test server");
        let req = format!(
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(req.as_bytes()).expect("write request");
        let mut resp = String::new();
        stream.read_to_string(&mut resp).expect("read response");
        resp
    }

    #[test]
    fn health_and_model_listing_endpoints_respond() {
        let addr = spawn_test_server();

        let health = send_raw_http(addr, "GET", "/", "");
        assert!(health.starts_with("HTTP/1.1 200 OK"));
        assert!(health.contains("aibox is running"));

        let openai_models = send_raw_http(addr, "GET", "/v1/models", "");
        assert!(openai_models.contains("\"object\":\"list\""));
        assert!(openai_models.contains("\"owned_by\":\"aibox\""));

        let ollama_tags = send_raw_http(addr, "GET", "/api/tags", "");
        assert!(ollama_tags.contains("\"models\":["));
    }

    #[test]
    fn openai_chat_completions_supports_sync_and_sse_streaming() {
        let addr = spawn_test_server();
        let sync_body = r#"{"model":"aibox-ternary-demo","messages":[{"role":"user","content":"Bonjour"}],"max_tokens":4,"stream":false}"#;
        let sync_resp = send_raw_http(addr, "POST", "/v1/chat/completions", sync_body);
        assert!(sync_resp.starts_with("HTTP/1.1 200 OK"));
        assert!(sync_resp.contains("\"object\":\"chat.completion\""));
        assert!(sync_resp.contains("\"completion_tokens\":4"));

        let stream_body = r#"{"model":"aibox-ternary-demo","messages":[{"role":"user","content":"Bonjour"}],"max_tokens":3,"stream":true}"#;
        let stream_resp = send_raw_http(addr, "POST", "/v1/chat/completions", stream_body);
        assert!(stream_resp.contains("Content-Type: text/event-stream"));
        assert!(stream_resp.contains("\"object\":\"chat.completion.chunk\""));
        assert!(stream_resp.contains("data: [DONE]"));
    }

    #[test]
    fn ollama_chat_and_generate_support_ndjson_and_sync() {
        let addr = spawn_test_server();
        let chat_body = r#"{"model":"aibox-ternary-demo","messages":[{"role":"user","content":"Salut"}],"options":{"num_predict":3},"stream":true}"#;
        let chat_resp = send_raw_http(addr, "POST", "/api/chat", chat_body);
        assert!(chat_resp.contains("Content-Type: application/x-ndjson"));
        assert!(chat_resp.contains("\"done\":false"));
        assert!(chat_resp.contains("\"done\":true"));
        assert!(chat_resp.contains("\"eval_count\":3"));

        let gen_body = r#"{"model":"aibox-ternary-demo","prompt":"Explique T-MAC","options":{"num_predict":2},"stream":false}"#;
        let gen_resp = send_raw_http(addr, "POST", "/api/generate", gen_body);
        assert!(gen_resp.starts_with("HTTP/1.1 200 OK"));
        assert!(gen_resp.contains("\"response\":\"[tmac:"));
        assert!(gen_resp.contains("\"eval_count\":2"));
    }
}
