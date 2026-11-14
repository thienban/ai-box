//! SERV-02 : Tokenizer BPE léger (sans dépendance externe) compatible Qwen3 / GGUF vocab.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::json::{parse_json, JsonValue};

pub struct SimpleTokenizer {
    id_to_piece: Vec<String>,
    piece_to_id: HashMap<String, usize>,
    max_piece_chars: usize,
    eos_id: Option<usize>,
}

impl SimpleTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let raw = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read '{}': {e}", path.display()))?;
        let root = parse_json(&raw)?;
        let root_obj = root
            .as_object()
            .ok_or_else(|| "tokenizer.json must be a JSON object".to_string())?;

        let vocab_obj = root_obj
            .get("model")
            .and_then(JsonValue::as_object)
            .and_then(|m| m.get("vocab"))
            .and_then(JsonValue::as_object)
            .ok_or_else(|| "Missing model.vocab in tokenizer.json".to_string())?;

        let mut max_id = 0usize;
        let mut piece_to_id = HashMap::with_capacity(vocab_obj.len());
        let mut max_piece_chars = 1usize;

        for (piece, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                let id = id as usize;
                max_id = max_id.max(id);
                max_piece_chars = max_piece_chars.max(piece.chars().count());
                piece_to_id.insert(piece.clone(), id);
            }
        }

        if let Some(added) = root_obj.get("added_tokens").and_then(JsonValue::as_array) {
            for item in added {
                if let Some(obj) = item.as_object() {
                    if let (Some(content), Some(id)) = (
                        obj.get("content").and_then(JsonValue::as_str),
                        obj.get("id").and_then(JsonValue::as_u64),
                    ) {
                        let id = id as usize;
                        max_id = max_id.max(id);
                        piece_to_id.insert(content.to_string(), id);
                    }
                }
            }
        }

        let mut id_to_piece = vec![String::new(); max_id + 1];
        for (piece, &id) in &piece_to_id {
            if id < id_to_piece.len() {
                id_to_piece[id] = piece.clone();
            }
        }

        let eos_id = piece_to_id
            .get("<|im_end|>")
            .or_else(|| piece_to_id.get("<|endoftext|>"))
            .copied();

        Ok(Self {
            id_to_piece,
            piece_to_id,
            max_piece_chars: max_piece_chars.min(64),
            eos_id,
        })
    }

    pub fn eos_token_id(&self) -> Option<usize> {
        self.eos_id
    }

    /// Encode un texte par longest-match glouton (gère `Ġ` espace et `Ċ` retour ligne).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let normalized: String = text
            .chars()
            .map(|ch| match ch {
                ' ' => '\u{0120}',
                '\n' => '\u{010A}',
                other => other,
            })
            .collect();

        let chars: Vec<char> = normalized.chars().collect();
        let mut ids = Vec::new();
        let mut idx = 0usize;

        while idx < chars.len() {
            let max_len = (chars.len() - idx).min(self.max_piece_chars);
            let mut matched = None;
            for len in (1..=max_len).rev() {
                let candidate: String = chars[idx..idx + len].iter().collect();
                if let Some(&id) = self.piece_to_id.get(&candidate) {
                    matched = Some((id, len));
                    break;
                }
            }

            if let Some((id, len)) = matched {
                ids.push(id);
                idx += len;
            } else {
                let byte_fallback = format!("<0x{:02X}>", chars[idx] as u32 & 0xFF);
                if let Some(&id) = self.piece_to_id.get(&byte_fallback) {
                    ids.push(id);
                }
                idx += 1;
            }
        }
        ids
    }

    pub fn encode_chat(&self, user_prompt: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        if let Some(&im_start) = self.piece_to_id.get("<|im_start|>") {
            ids.push(im_start);
            ids.extend(self.encode("user\n"));
            ids.extend(self.encode(user_prompt));
            if let Some(&im_end) = self.piece_to_id.get("<|im_end|>") {
                ids.push(im_end);
            }
            ids.extend(self.encode("\n"));
            ids.push(im_start);
            ids.extend(self.encode("assistant\n"));
        } else {
            ids = self.encode(user_prompt);
        }
        ids
    }

    pub fn decode_token(&self, token_id: usize) -> String {
        let piece = self.id_to_piece.get(token_id).map(String::as_str).unwrap_or("");
        if piece == "<|im_end|>" || piece == "<|endoftext|>" || piece == "<|im_start|>" {
            return String::new();
        }
        piece
            .chars()
            .map(|ch| match ch {
                '\u{0120}' => ' ',
                '\u{010A}' => '\n',
                other => other,
            })
            .collect()
    }

    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in token_ids {
            out.push_str(&self.decode_token(id));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn encodes_and_decodes_chat_prompt() {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = env::temp_dir().join(format!("aibox_tok_{unique}.json"));
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {
                    "<|im_start|>": 0,
                    "<|im_end|>": 1,
                    "user": 2,
                    "assistant": 3,
                    "Ċ": 4,
                    "Bon": 5,
                    "jour": 6,
                    "ĠParis": 7
                }
            }
        }"#;
        fs::write(&path, json).unwrap();

        let tok = SimpleTokenizer::from_file(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(tok.eos_token_id(), Some(1));
        let ids = tok.encode_chat("Bonjour Paris");
        assert_eq!(ids, vec![0, 2, 4, 5, 6, 7, 1, 4, 0, 3, 4]);
        assert_eq!(tok.decode(&[5, 6, 7]), "Bonjour Paris");
    }
}
