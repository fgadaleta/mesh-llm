#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::frontend) enum NativeMtpTrimAction {
    None,
    FullSession,
}

pub(in crate::frontend) fn native_mtp_trim_action(
    committed_positions: usize,
    consumed_positions: usize,
) -> NativeMtpTrimAction {
    if committed_positions < consumed_positions {
        NativeMtpTrimAction::FullSession
    } else {
        NativeMtpTrimAction::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_or_partially_committed_spans_require_full_session_trim() {
        assert_eq!(
            native_mtp_trim_action(0, 2),
            NativeMtpTrimAction::FullSession
        );
        assert_eq!(
            native_mtp_trim_action(1, 2),
            NativeMtpTrimAction::FullSession
        );
    }

    #[test]
    fn fully_committed_spans_do_not_trim() {
        assert_eq!(native_mtp_trim_action(2, 2), NativeMtpTrimAction::None);
        assert_eq!(native_mtp_trim_action(3, 2), NativeMtpTrimAction::None);
    }
}
