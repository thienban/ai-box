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
    is_sentencepiece: bool,
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
            .or_else(|| piece_to_id.get("<turn|>"))
            .or_else(|| piece_to_id.get("<|endoftext|>"))
            .or_else(|| piece_to_id.get("<eos>"))
            .copied();

        let is_sentencepiece = !piece_to_id.contains_key("Ġthe")
            && (piece_to_id.contains_key("<bos>") || piece_to_id.keys().any(|k| k.starts_with('▁')));

        Ok(Self {
            id_to_piece,
            piece_to_id,
            max_piece_chars: max_piece_chars.min(64),
            eos_id,
            is_sentencepiece,
        })
    }

    pub fn eos_token_id(&self) -> Option<usize> {
        self.eos_id
    }

    /// Encode un texte par longest-match glouton (GPT-2 byte-level ou SentencePiece `▁`).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let chars: Vec<char> = if self.is_sentencepiece {
            text.chars().map(|c| if c == ' ' { '▁' } else { c }).collect()
        } else {
            text.as_bytes().iter().copied().map(byte_to_gpt2_char).collect()
        };
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
        } else if let Some(&turn_start) = self.piece_to_id.get("<|turn>") {
            if let Some(&bos) = self.piece_to_id.get("<bos>") {
                ids.push(bos);
            }
            ids.push(turn_start);
            ids.extend(self.encode("user\n"));
            ids.extend(self.encode(user_prompt));
            if let Some(&turn_end) = self.piece_to_id.get("<turn|>") {
                ids.push(turn_end);
            }
            ids.extend(self.encode("\n"));
            ids.push(turn_start);
            ids.extend(self.encode("model\n"));
        } else {
            ids = self.encode(user_prompt);
        }
        ids
    }

    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut decoder = TokenStreamDecoder::new(self);
        let mut out = String::new();
        for &id in token_ids {
            out.push_str(&decoder.push(id));
        }
        out.push_str(&decoder.finish());
        out
    }

    fn raw_piece(&self, token_id: usize) -> &str {
        self.id_to_piece.get(token_id).map(String::as_str).unwrap_or("")
    }
}

/// Décodeur incrémental gérant les octets UTF-8 GPT-2/Qwen répartis sur plusieurs tokens.
pub struct TokenStreamDecoder<'a> {
    tokenizer: &'a SimpleTokenizer,
    pending_bytes: Vec<u8>,
}

impl<'a> TokenStreamDecoder<'a> {
    pub fn new(tokenizer: &'a SimpleTokenizer) -> Self {
        Self {
            tokenizer,
            pending_bytes: Vec::new(),
        }
    }

    pub fn push(&mut self, token_id: usize) -> String {
        let piece = self.tokenizer.raw_piece(token_id);
        if matches!(
            piece,
            "<|im_end|>" | "<|endoftext|>" | "<|im_start|>" | "<bos>" | "<eos>" | "<|turn>" | "<turn|>"
        ) {
            return String::new();
        }
        if self.tokenizer.is_sentencepiece {
            return piece.replace('▁', " ");
        }
        for ch in piece.chars() {
            if let Some(byte) = gpt2_unicode_to_byte(ch) {
                self.pending_bytes.push(byte);
            } else {
                let mut buf = [0u8; 4];
                self.pending_bytes
                    .extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        self.drain_valid_utf8()
    }

    pub fn finish(&mut self) -> String {
        if self.pending_bytes.is_empty() {
            return String::new();
        }
        let lossy = String::from_utf8_lossy(&self.pending_bytes).to_string();
        self.pending_bytes.clear();
        lossy
    }

    fn drain_valid_utf8(&mut self) -> String {
        match std::str::from_utf8(&self.pending_bytes) {
            Ok(valid) => {
                let s = valid.to_string();
                self.pending_bytes.clear();
                s
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                if valid_up_to == 0 {
                    if err.error_len().is_none() {
                        return String::new();
                    }
                    let lossy = String::from_utf8_lossy(&self.pending_bytes).to_string();
                    self.pending_bytes.clear();
                    return lossy;
                }
                let s = std::str::from_utf8(&self.pending_bytes[..valid_up_to])
                    .unwrap_or("")
                    .to_string();
                self.pending_bytes.drain(..valid_up_to);
                s
            }
        }
    }
}

fn gpt2_unicode_to_byte(ch: char) -> Option<u8> {
    let code = ch as u32;
    if (33..=126).contains(&code) || (161..=172).contains(&code) || (174..=255).contains(&code) {
        return Some(code as u8);
    }
    if !(256..324).contains(&code) {
        return None;
    }
    let mut target = 256u32;
    for b in 0u32..=255 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        if !printable {
            if target == code {
                return Some(b as u8);
            }
            target += 1;
        }
    }
    None
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

    #[test]
    fn decodes_gpt2_byte_level_utf8_emoji() {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = env::temp_dir().join(format!("aibox_tok_utf8_{unique}.json"));
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {
                    "Bonjour": 0,
                    "Ġ!": 1,
                    "Ġ": 2,
                    "ðŁĺĬ": 3
                }
            }
        }"#;
        fs::write(&path, json).unwrap();
        let tok = SimpleTokenizer::from_file(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(tok.decode(&[0, 1, 2, 3]), "Bonjour ! 😊");
    }
}

const GPT2_BYTE_LUT: [char; 256] = {
    let mut lut = ['\0'; 256];
    let mut extra = 0u32;
    let mut byte = 0usize;
    while byte < 256 {
        let is_direct = (byte >= 33 && byte <= 126)
            || (byte >= 161 && byte <= 172)
            || (byte >= 174 && byte <= 255);
        if is_direct {
            lut[byte] = char::from_u32(byte as u32).unwrap();
        } else {
            lut[byte] = char::from_u32(256 + extra).unwrap();
            extra += 1;
        }
        byte += 1;
    }
    lut
};

#[inline]
fn byte_to_gpt2_char(b: u8) -> char {
    GPT2_BYTE_LUT[b as usize]
}
