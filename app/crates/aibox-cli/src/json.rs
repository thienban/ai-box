//! Parseur JSON minimaliste sans dépendance externe pour `safetensors` et `quant_config.json`.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(map) => Some(map),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(n) if *n >= 0.0 => Some(*n as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Échappe une chaîne pour l'inclure en sécurité dans un littéral JSON (avec guillemets).
pub fn escape_json_string(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    out.push('"');
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn parse_json(input: &str) -> Result<JsonValue, String> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    let val = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(format!("Unexpected trailing characters at byte {}", parser.pos));
    }
    Ok(val)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next_byte(&mut self) -> Result<u8, String> {
        let b = self
            .peek()
            .ok_or_else(|| "Unexpected end of JSON".to_string())?;
        self.pos += 1;
        Ok(b)
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b't') => self.expect_literal(b"true", JsonValue::Bool(true)),
            Some(b'f') => self.expect_literal(b"false", JsonValue::Bool(false)),
            Some(b'n') => self.expect_literal(b"null", JsonValue::Null),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(ch) => Err(format!("Unexpected byte '{}' at pos {}", ch as char, self.pos)),
            None => Err("Unexpected end of JSON".to_string()),
        }
    }

    fn expect_literal(&mut self, lit: &[u8], value: JsonValue) -> Result<JsonValue, String> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            return Ok(value);
        }
        Err(format!("Invalid literal at pos {}", self.pos))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.next_byte()? != b'"' {
            return Err("Expected '\"'".to_string());
        }
        let mut out = String::new();
        loop {
            let b = self.next_byte()?;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = self.next_byte()?;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        _ => out.push(esc as char),
                    }
                }
                other => out.push(other as char),
            }
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') = self.peek() {
            self.pos += 1;
        }
        let slice = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|e| format!("Invalid UTF-8 in number: {e}"))?;
        let num = slice
            .parse::<f64>()
            .map_err(|e| format!("Invalid number '{slice}': {e}"))?;
        Ok(JsonValue::Number(num))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.next_byte()?; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.next_byte()? {
                b',' => continue,
                b']' => return Ok(JsonValue::Array(items)),
                other => return Err(format!("Expected ',' or ']', got '{}'", other as char)),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.next_byte()?; // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.next_byte()? != b':' {
                return Err("Expected ':' in object".to_string());
            }
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.next_byte()? {
                b',' => continue,
                b'}' => return Ok(JsonValue::Object(map)),
                other => return Err(format!("Expected ',' or '}}', got '{}'", other as char)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_safetensors_style_header() {
        let raw = r#"{"w":{"dtype":"U8","shape":[4,32],"data_offsets":[0,128]},"__metadata__":{"format":"pt"}}"#;
        let val = parse_json(raw).expect("valid json");
        let obj = val.as_object().unwrap();
        let w = obj.get("w").unwrap().as_object().unwrap();
        assert_eq!(w.get("dtype").unwrap().as_str(), Some("U8"));
        assert_eq!(w.get("shape").unwrap().as_array().unwrap().len(), 2);
    }
}
