const BATCHED_VERIFY_ENV: &str = "SKIPPY_NATIVE_MTP_BATCHED_VERIFY";
const REJECT_COOLDOWN_TOKENS_ENV: &str = "SKIPPY_NATIVE_MTP_REJECT_COOLDOWN_TOKENS";
const DEFER_REJECT_TRIM_ENV: &str = "SKIPPY_NATIVE_MTP_DEFER_REJECT_TRIM";
const SUPPRESS_COOLDOWN_DRAFTS_ENV: &str = "SKIPPY_NATIVE_MTP_SUPPRESS_COOLDOWN_DRAFTS";
const SUPPRESS_COOLDOWN_DRAFT_LIMIT_ENV: &str = "SKIPPY_NATIVE_MTP_SUPPRESS_COOLDOWN_DRAFT_LIMIT";

pub(in crate::frontend) fn native_mtp_batched_verify_enabled() -> bool {
    native_mtp_batched_verify_enabled_from(std::env::var(BATCHED_VERIFY_ENV).ok().as_deref())
}

pub(in crate::frontend) fn native_mtp_reject_cooldown_tokens() -> usize {
    parse_usize_env(REJECT_COOLDOWN_TOKENS_ENV, 0)
}

pub(in crate::frontend) fn native_mtp_defer_reject_trim_enabled() -> bool {
    truthy_env(std::env::var(DEFER_REJECT_TRIM_ENV).ok().as_deref())
}

pub(in crate::frontend) fn native_mtp_suppress_cooldown_drafts_enabled() -> bool {
    truthy_env(std::env::var(SUPPRESS_COOLDOWN_DRAFTS_ENV).ok().as_deref())
}

pub(in crate::frontend) fn native_mtp_suppress_cooldown_draft_limit() -> usize {
    parse_usize_env(SUPPRESS_COOLDOWN_DRAFT_LIMIT_ENV, 0)
}

fn native_mtp_batched_verify_enabled_from(value: Option<&str>) -> bool {
    !falsey_env(value)
}

fn truthy_env(value: Option<&str>) -> bool {
    matches!(
        normalized_env(value).as_deref(),
        Some("1" | "true" | "on" | "enable" | "enabled" | "yes")
    )
}

fn falsey_env(value: Option<&str>) -> bool {
    matches!(
        normalized_env(value).as_deref(),
        Some("0" | "false" | "off" | "disable" | "disabled" | "no")
    )
}

fn normalized_env(value: Option<&str>) -> Option<String> {
    value.map(str::trim).map(str::to_ascii_lowercase)
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_verify_flag_defaults_on_and_accepts_false_values() {
        assert!(native_mtp_batched_verify_enabled_from(None));
        assert!(native_mtp_batched_verify_enabled_from(Some("1")));
        assert!(native_mtp_batched_verify_enabled_from(Some("true")));
        assert!(!native_mtp_batched_verify_enabled_from(Some("0")));
        assert!(!native_mtp_batched_verify_enabled_from(Some("false")));
        assert!(!native_mtp_batched_verify_enabled_from(Some(" disabled ")));
    }

    #[test]
    fn truthy_env_accepts_enabled_aliases_only() {
        for value in ["1", "true", " enabled ", "yes", "on"] {
            assert!(truthy_env(Some(value)), "{value}");
        }
        for value in [
            None,
            Some("0"),
            Some("false"),
            Some("off"),
            Some("disabled"),
        ] {
            assert!(!truthy_env(value), "{value:?}");
        }
    }

    #[test]
    fn numeric_options_default_when_absent() {
        assert_eq!(parse_usize_env("SKIPPY_TEST_MISSING_REJECT_COOLDOWN", 0), 0);
        assert_eq!(
            parse_usize_env("SKIPPY_TEST_MISSING_SUPPRESS_COOLDOWN_LIMIT", 0),
            0
        );
    }
}
