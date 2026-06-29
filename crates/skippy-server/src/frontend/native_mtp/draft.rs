#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpDraft {
    pub(in crate::frontend) token: i32,
    pub(in crate::frontend) proposal_compute_us: i64,
}

impl NativeMtpDraft {
    pub(in crate::frontend) fn from_prediction_tokens(tokens: &[i32]) -> Option<Self> {
        let token = *tokens.get(1)?;
        let proposal_compute_us = tokens.get(2).copied().unwrap_or_default();
        Some(Self {
            token,
            proposal_compute_us: i64::from(proposal_compute_us.max(0)),
        })
    }

    pub(in crate::frontend) fn from_verify_prediction_tokens(
        tokens: &[i32],
        verified_token_count: usize,
    ) -> Option<Self> {
        let token = *tokens.get(verified_token_count)?;
        let proposal_compute_us = tokens
            .get(verified_token_count.saturating_add(1))
            .copied()
            .unwrap_or_default();
        Some(Self {
            token,
            proposal_compute_us: i64::from(proposal_compute_us.max(0)),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::frontend) struct PendingNativeMtpDraft {
    pub(in crate::frontend) token: i32,
    pub(in crate::frontend) origin: NativeMtpDraftOrigin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::frontend) enum NativeMtpDraftOrigin {
    InitialSerial,
    SerialAfterGap,
    VerifyNext,
}

impl NativeMtpDraftOrigin {
    pub(in crate::frontend) fn label(self) -> &'static str {
        match self {
            Self::InitialSerial => "initial_serial",
            Self::SerialAfterGap => "serial_after_gap",
            Self::VerifyNext => "verify_next",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prediction_token_sideband() {
        assert_eq!(
            NativeMtpDraft::from_prediction_tokens(&[11, 12, 34]),
            Some(NativeMtpDraft {
                token: 12,
                proposal_compute_us: 34,
            })
        );
        assert_eq!(
            NativeMtpDraft::from_prediction_tokens(&[11, 12, 34, 567]),
            Some(NativeMtpDraft {
                token: 12,
                proposal_compute_us: 34,
            })
        );
        assert_eq!(NativeMtpDraft::from_prediction_tokens(&[11]), None);
    }

    #[test]
    fn parses_verify_prediction_token_sideband_after_verified_tokens() {
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11, 12, 34], 2),
            Some(NativeMtpDraft {
                token: 12,
                proposal_compute_us: 34,
            })
        );
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11, 12, -3], 2),
            Some(NativeMtpDraft {
                token: 12,
                proposal_compute_us: 0,
            })
        );
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11, 12, 34, 567], 2),
            Some(NativeMtpDraft {
                token: 12,
                proposal_compute_us: 34,
            })
        );
        assert_eq!(
            NativeMtpDraft::from_verify_prediction_tokens(&[10, 11], 2),
            None
        );
    }

    #[test]
    fn pending_draft_keeps_origin_label() {
        let pending = PendingNativeMtpDraft {
            token: 12,
            origin: NativeMtpDraftOrigin::VerifyNext,
        };

        assert_eq!(pending.token, 12);
        assert_eq!(pending.origin.label(), "verify_next");
    }
}
