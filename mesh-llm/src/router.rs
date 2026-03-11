/// Smart model router — classifies requests and picks the best model.

use serde_json::Value;

// ── Request categories ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Code,
    Reasoning,
    Chat,
    ToolCall,
    Creative,
}

// ── Model profiles ──────────────────────────────────────────────────

/// Quality tier: higher = better quality, slower.
/// 1 = draft/tiny, 2 = good, 3 = strong, 4 = frontier
pub type Tier = u8;

pub struct ModelProfile {
    pub name: &'static str,
    pub strengths: &'static [Category],
    pub tier: Tier,
}

/// Static profiles for catalog models.
/// Order of strengths matters — first entry is primary strength.
pub static MODEL_PROFILES: &[ModelProfile] = &[
    // ── Tier 4: Frontier ────────────────────────────────────────
    ModelProfile {
        name: "Qwen3-235B-A22B-Q4_K_M",
        strengths: &[Category::Code, Category::Reasoning, Category::Chat, Category::Creative],
        tier: 4,
    },
    ModelProfile {
        name: "Llama-3.1-405B-Instruct-Q2_K",
        strengths: &[Category::Chat, Category::Reasoning, Category::Code],
        tier: 4,
    },
    ModelProfile {
        name: "MiniMax-M2.5-Q4_K_M",
        strengths: &[Category::Chat, Category::Creative, Category::Reasoning],
        tier: 4,
    },
    // ── Tier 3: Strong ──────────────────────────────────────────
    ModelProfile {
        name: "Qwen2.5-72B-Instruct-Q4_K_M",
        strengths: &[Category::Chat, Category::Reasoning, Category::Code],
        tier: 3,
    },
    ModelProfile {
        name: "Llama-3.3-70B-Instruct-Q4_K_M",
        strengths: &[Category::Chat, Category::ToolCall, Category::Code],
        tier: 3,
    },
    ModelProfile {
        name: "DeepSeek-R1-Distill-70B-Q4_K_M",
        strengths: &[Category::Reasoning],
        tier: 3,
    },
    ModelProfile {
        name: "Mixtral-8x22B-Instruct-Q4_K_M",
        strengths: &[Category::Chat, Category::Code, Category::Reasoning],
        tier: 3,
    },
    ModelProfile {
        name: "Qwen3-32B-Q4_K_M",
        strengths: &[Category::Reasoning, Category::Code, Category::Chat],
        tier: 3,
    },
    ModelProfile {
        name: "Qwen2.5-Coder-32B-Instruct-Q4_K_M",
        strengths: &[Category::Code],
        tier: 3,
    },
    ModelProfile {
        name: "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M",
        strengths: &[Category::Reasoning],
        tier: 3,
    },
    ModelProfile {
        name: "Qwen3-30B-A3B-Q4_K_M",
        strengths: &[Category::Chat, Category::Reasoning, Category::Code],
        tier: 3,
    },
    ModelProfile {
        name: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M",
        strengths: &[Category::Code, Category::ToolCall],
        tier: 3,
    },
    ModelProfile {
        name: "Qwen2.5-32B-Instruct-Q4_K_M",
        strengths: &[Category::Chat, Category::Reasoning],
        tier: 3,
    },
    ModelProfile {
        name: "Gemma-3-27B-it-Q4_K_M",
        strengths: &[Category::Reasoning, Category::Chat],
        tier: 3,
    },
    // ── Tier 2: Good ────────────────────────────────────────────
    ModelProfile {
        name: "Mistral-Small-3.1-24B-Instruct-Q4_K_M",
        strengths: &[Category::Chat, Category::ToolCall],
        tier: 2,
    },
    ModelProfile {
        name: "Devstral-Small-2505-Q4_K_M",
        strengths: &[Category::Code, Category::ToolCall],
        tier: 2,
    },
    ModelProfile {
        name: "GLM-4.7-Flash-Q4_K_M",
        strengths: &[Category::Chat, Category::ToolCall],
        tier: 2,
    },
    ModelProfile {
        name: "GLM-4-32B-0414-Q4_K_M",
        strengths: &[Category::Chat, Category::ToolCall, Category::Code],
        tier: 2,
    },
    ModelProfile {
        name: "Llama-4-Scout-Q4_K_M",
        strengths: &[Category::Chat, Category::ToolCall],
        tier: 2,
    },
    ModelProfile {
        name: "Qwen3-14B-Q4_K_M",
        strengths: &[Category::Chat, Category::Reasoning],
        tier: 2,
    },
    ModelProfile {
        name: "Qwen2.5-14B-Instruct-Q4_K_M",
        strengths: &[Category::Chat],
        tier: 2,
    },
    ModelProfile {
        name: "Qwen2.5-Coder-14B-Instruct-Q4_K_M",
        strengths: &[Category::Code],
        tier: 2,
    },
    ModelProfile {
        name: "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M",
        strengths: &[Category::Reasoning],
        tier: 2,
    },
    ModelProfile {
        name: "Gemma-3-12B-it-Q4_K_M",
        strengths: &[Category::Chat, Category::Reasoning],
        tier: 2,
    },
    ModelProfile {
        name: "Qwen3-8B-Q4_K_M",
        strengths: &[Category::Chat, Category::Code],
        tier: 2,
    },
    ModelProfile {
        name: "Hermes-2-Pro-Mistral-7B-Q4_K_M",
        strengths: &[Category::ToolCall, Category::Chat],
        tier: 2,
    },
    ModelProfile {
        name: "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
        strengths: &[Category::Code],
        tier: 2,
    },
    // ── Tier 1: Small / Draft ───────────────────────────────────
    ModelProfile {
        name: "Qwen3-4B-Q4_K_M",
        strengths: &[Category::Chat],
        tier: 1,
    },
    ModelProfile {
        name: "Qwen2.5-3B-Instruct-Q4_K_M",
        strengths: &[Category::Chat],
        tier: 1,
    },
    ModelProfile {
        name: "Llama-3.2-3B-Instruct-Q4_K_M",
        strengths: &[Category::Chat, Category::ToolCall],
        tier: 1,
    },
];

pub fn profile_for(model_name: &str) -> Option<&'static ModelProfile> {
    MODEL_PROFILES.iter().find(|p| p.name == model_name)
}

// ── Request classification ──────────────────────────────────────────

/// Classify a chat completion request body using heuristics.
/// No LLM call, just pattern matching on the request structure.
pub fn classify(body: &Value) -> Category {
    // 1. Has tools → tool_call
    if let Some(tools) = body.get("tools") {
        if tools.is_array() && !tools.as_array().unwrap().is_empty() {
            return Category::ToolCall;
        }
    }

    // Collect all text from messages for keyword analysis
    let text = collect_message_text(body);
    let lower = text.to_lowercase();

    // 2. Code signals
    let code_signals = [
        "```", "def ", "fn ", "func ", "class ", "import ",
        "function ", "const ", "let ", "var ", "return ",
        "write a program", "write code", "implement", "refactor",
        "debug", "fix the bug", "write a script", "code review",
        "pull request", "git ", "compile", "syntax",
    ];
    let code_score: usize = code_signals.iter()
        .filter(|s| lower.contains(*s))
        .count();

    // 3. Reasoning signals
    let reasoning_signals = [
        "prove", "explain why", "step by step", "calculate",
        "solve", "derive", "what is the probability", "how many",
        "analyze", "compare and contrast", "evaluate",
        "mathematical", "theorem", "equation", "logic",
        "think carefully", "reason about",
    ];
    let reasoning_score: usize = reasoning_signals.iter()
        .filter(|s| lower.contains(*s))
        .count();

    // 4. Creative signals
    let creative_signals = [
        "write a story", "write a poem", "creative", "imagine",
        "fiction", "narrative", "compose", "brainstorm",
        "write a song", "screenplay", "dialogue",
    ];
    let creative_score: usize = creative_signals.iter()
        .filter(|s| lower.contains(*s))
        .count();

    // 5. System prompt hints
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            if msg.get("role").and_then(|r| r.as_str()) == Some("system") {
                if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                    let sys = content.to_lowercase();
                    if sys.contains("developer") || sys.contains("coding") || sys.contains("programmer") {
                        return Category::Code;
                    }
                }
            }
        }
    }

    // Pick highest scoring category
    if code_score >= 2 || (code_score >= 1 && reasoning_score == 0 && creative_score == 0) {
        Category::Code
    } else if reasoning_score >= 2 {
        Category::Reasoning
    } else if creative_score >= 1 {
        Category::Creative
    } else {
        Category::Chat
    }
}

fn collect_message_text(body: &Value) -> String {
    let mut text = String::new();
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                text.push_str(content);
                text.push('\n');
            }
        }
    }
    text
}

// ── Model selection ─────────────────────────────────────────────────

/// Pick the best model from available models for a given category.
/// Returns the model name. Falls back to highest-tier available model
/// if nothing matches the category specifically.
pub fn pick_model<'a>(
    category: Category,
    available_models: &[(&'a str, f64)], // (model_name, observed_tok_per_sec)
) -> Option<&'a str> {
    if available_models.is_empty() {
        return None;
    }

    // Score each available model for this category
    let mut scored: Vec<(&str, i32)> = available_models
        .iter()
        .map(|(name, tok_s)| {
            let profile = profile_for(name);
            let tier = profile.map(|p| p.tier).unwrap_or(1) as i32;

            // Strength match bonus: primary strength = +20, secondary = +10, any match = +5
            let strength_bonus = profile
                .map(|p| {
                    p.strengths
                        .iter()
                        .enumerate()
                        .find(|(_, s)| **s == category)
                        .map(|(i, _)| match i {
                            0 => 20,
                            1 => 10,
                            _ => 5,
                        })
                        .unwrap_or(0)
                })
                .unwrap_or(0);

            // Speed bonus: normalize tok/s to 0-10 range (100+ tok/s = max bonus)
            let speed_bonus = (tok_s / 10.0).min(10.0) as i32;

            let score = tier * 10 + strength_bonus + speed_bonus;
            (*name, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.cmp(&a.1));
    scored.first().map(|(name, _)| *name)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_classify_tool_call() {
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}]
        });
        assert_eq!(classify(&body), Category::ToolCall);
    }

    #[test]
    fn test_classify_code() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Write a Python function to implement binary search and debug any issues"}
            ]
        });
        assert_eq!(classify(&body), Category::Code);
    }

    #[test]
    fn test_classify_reasoning() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Prove that the square root of 2 is irrational. Explain step by step."}
            ]
        });
        assert_eq!(classify(&body), Category::Reasoning);
    }

    #[test]
    fn test_classify_creative() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Write a story about a robot who learns to paint"}
            ]
        });
        assert_eq!(classify(&body), Category::Creative);
    }

    #[test]
    fn test_classify_chat_default() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "What's the capital of France?"}
            ]
        });
        assert_eq!(classify(&body), Category::Chat);
    }

    #[test]
    fn test_classify_system_prompt_code() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "You are a senior developer and coding assistant."},
                {"role": "user", "content": "Help me with this."}
            ]
        });
        assert_eq!(classify(&body), Category::Code);
    }

    #[test]
    fn test_pick_model_prefers_tier() {
        let available = vec![
            ("Qwen3-8B-Q4_K_M", 50.0),
            ("Qwen3-235B-A22B-Q4_K_M", 20.0),
        ];
        let result = pick_model(Category::Chat, &available);
        assert_eq!(result, Some("Qwen3-235B-A22B-Q4_K_M"));
    }

    #[test]
    fn test_pick_model_prefers_strength_match() {
        let available = vec![
            ("DeepSeek-R1-Distill-70B-Q4_K_M", 10.0), // tier 3, reasoning specialist
            ("Qwen2.5-72B-Instruct-Q4_K_M", 10.0),     // tier 3, chat primary
        ];
        let result = pick_model(Category::Reasoning, &available);
        assert_eq!(result, Some("DeepSeek-R1-Distill-70B-Q4_K_M"));
    }

    #[test]
    fn test_pick_model_code_specialist() {
        let available = vec![
            ("Qwen2.5-Coder-32B-Instruct-Q4_K_M", 15.0),
            ("Qwen2.5-32B-Instruct-Q4_K_M", 15.0),
        ];
        let result = pick_model(Category::Code, &available);
        assert_eq!(result, Some("Qwen2.5-Coder-32B-Instruct-Q4_K_M"));
    }

    #[test]
    fn test_pick_model_empty() {
        let available: Vec<(&str, f64)> = vec![];
        assert_eq!(pick_model(Category::Chat, &available), None);
    }

    #[test]
    fn test_pick_model_unknown_model_still_works() {
        let available = vec![("SomeUnknownModel", 30.0)];
        let result = pick_model(Category::Chat, &available);
        assert_eq!(result, Some("SomeUnknownModel"));
    }

    #[test]
    fn test_profile_lookup() {
        assert!(profile_for("Qwen3-235B-A22B-Q4_K_M").is_some());
        assert_eq!(profile_for("Qwen3-235B-A22B-Q4_K_M").unwrap().tier, 4);
        assert!(profile_for("nonexistent").is_none());
    }

    #[test]
    fn test_all_profiles_have_strengths() {
        for p in MODEL_PROFILES {
            assert!(!p.strengths.is_empty(), "{} has no strengths", p.name);
        }
    }

    #[test]
    fn test_classify_empty_tools_is_not_tool_call() {
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": []
        });
        assert_eq!(classify(&body), Category::Chat);
    }
}
