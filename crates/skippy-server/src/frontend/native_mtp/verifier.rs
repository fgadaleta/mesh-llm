use super::{
    NativeMtpDraft, NativeMtpDraftOrigin, NativeMtpN1Stats, NativeMtpVerification,
    PendingNativeMtpDraft,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingDraft {
    token: i32,
    origin: NativeMtpDraftOrigin,
}

#[derive(Default)]
pub(in crate::frontend) struct NativeMtpN1Verifier {
    pending: Option<PendingDraft>,
    stats: NativeMtpN1Stats,
}

impl NativeMtpN1Verifier {
    pub(in crate::frontend) fn take_pending_draft(&mut self) -> Option<PendingNativeMtpDraft> {
        self.pending.take().map(|pending| PendingNativeMtpDraft {
            token: pending.token,
            origin: pending.origin,
        })
    }

    pub(in crate::frontend) fn clear_pending_draft(&mut self) {
        self.pending = None;
    }

    pub(in crate::frontend) fn observe_taken_draft_verification(
        &mut self,
        draft_token: i32,
        target_token: i32,
        verification_compute_us: i64,
    ) -> NativeMtpVerification {
        self.record_verification(draft_token, target_token, verification_compute_us)
    }

    pub(in crate::frontend) fn observe_target_token(
        &mut self,
        target_token: i32,
        verification_compute_us: i64,
        next_draft: Option<NativeMtpDraft>,
        next_draft_origin: NativeMtpDraftOrigin,
    ) -> NativeMtpVerification {
        let verification = self.verify_pending(target_token, verification_compute_us);
        self.observe_next_draft(next_draft, next_draft_origin);
        verification
    }

    pub(in crate::frontend) fn stats(&self) -> NativeMtpN1Stats {
        let mut stats = self.stats;
        stats.pending_tokens = u64::from(self.pending.is_some());
        stats
    }

    fn verify_pending(
        &mut self,
        target_token: i32,
        verification_compute_us: i64,
    ) -> NativeMtpVerification {
        let Some(pending) = self.pending.take() else {
            return NativeMtpVerification::NoPending;
        };

        self.record_verification(pending.token, target_token, verification_compute_us)
    }

    fn record_verification(
        &mut self,
        draft_token: i32,
        target_token: i32,
        verification_compute_us: i64,
    ) -> NativeMtpVerification {
        self.stats.verification_count += 1;
        self.stats.verification_compute_us = self
            .stats
            .verification_compute_us
            .saturating_add(verification_compute_us);
        if draft_token == target_token {
            self.stats.accepted_tokens += 1;
            NativeMtpVerification::Accepted {
                draft: draft_token,
                target: target_token,
            }
        } else {
            self.stats.rejected_tokens += 1;
            NativeMtpVerification::Rejected {
                draft: draft_token,
                target: target_token,
            }
        }
    }

    pub(in crate::frontend) fn observe_next_draft(
        &mut self,
        next_draft: Option<NativeMtpDraft>,
        origin: NativeMtpDraftOrigin,
    ) {
        let Some(next_draft) = next_draft else {
            return;
        };
        self.stats.drafted_tokens += 1;
        self.stats.proposal_compute_us = self
            .stats
            .proposal_compute_us
            .saturating_add(next_draft.proposal_compute_us);
        self.pending = Some(PendingDraft {
            token: next_draft.token,
            origin,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(token: i32) -> NativeMtpDraft {
        NativeMtpDraft {
            token,
            proposal_compute_us: 7,
        }
    }

    fn observe(
        verifier: &mut NativeMtpN1Verifier,
        target_token: i32,
        verification_compute_us: i64,
        next_draft: Option<NativeMtpDraft>,
    ) -> NativeMtpVerification {
        verifier.observe_target_token(
            target_token,
            verification_compute_us,
            next_draft,
            NativeMtpDraftOrigin::InitialSerial,
        )
    }

    #[test]
    fn no_draft_behaves_like_baseline() {
        let mut verifier = NativeMtpN1Verifier::default();

        let decision = observe(&mut verifier, 11, 5, None);

        assert_eq!(decision, NativeMtpVerification::NoPending);
        assert_eq!(verifier.stats(), NativeMtpN1Stats::default());
    }

    #[test]
    fn first_draft_is_pending_until_next_target_decode() {
        let mut verifier = NativeMtpN1Verifier::default();

        let decision = observe(&mut verifier, 11, 5, Some(draft(12)));

        assert_eq!(decision, NativeMtpVerification::NoPending);
        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 1,
                pending_tokens: 1,
                proposal_compute_us: 7,
                ..NativeMtpN1Stats::default()
            }
        );
    }

    #[test]
    fn matching_next_target_accepts_pending_draft() {
        let mut verifier = NativeMtpN1Verifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let decision = observe(&mut verifier, 12, 9, None);

        assert_eq!(
            decision,
            NativeMtpVerification::Accepted {
                draft: 12,
                target: 12,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 1,
                accepted_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 7,
                verification_compute_us: 9,
                ..NativeMtpN1Stats::default()
            }
        );
    }

    #[test]
    fn different_next_target_rejects_pending_draft() {
        let mut verifier = NativeMtpN1Verifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let decision = observe(&mut verifier, 13, 9, None);

        assert_eq!(
            decision,
            NativeMtpVerification::Rejected {
                draft: 12,
                target: 13,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 1,
                rejected_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 7,
                verification_compute_us: 9,
                ..NativeMtpN1Stats::default()
            }
        );
    }

    #[test]
    fn verifies_previous_draft_before_storing_next_draft() {
        let mut verifier = NativeMtpN1Verifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let decision = observe(&mut verifier, 12, 9, Some(draft(14)));

        assert_eq!(
            decision,
            NativeMtpVerification::Accepted {
                draft: 12,
                target: 12,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 2,
                accepted_tokens: 1,
                pending_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 14,
                verification_compute_us: 9,
                ..NativeMtpN1Stats::default()
            }
        );
    }

    #[test]
    fn taken_pending_draft_can_be_recorded_as_batched_accept() {
        let mut verifier = NativeMtpN1Verifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let pending = verifier.take_pending_draft().unwrap();
        assert_eq!(pending.origin, NativeMtpDraftOrigin::InitialSerial);
        assert!(verifier.take_pending_draft().is_none());
        let decision = verifier.observe_taken_draft_verification(pending.token, 12, 9);

        assert_eq!(
            decision,
            NativeMtpVerification::Accepted {
                draft: 12,
                target: 12,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 1,
                accepted_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 7,
                verification_compute_us: 9,
                ..NativeMtpN1Stats::default()
            }
        );
    }

    #[test]
    fn taken_pending_draft_can_be_recorded_as_batched_reject() {
        let mut verifier = NativeMtpN1Verifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let pending = verifier.take_pending_draft().unwrap();
        assert_eq!(pending.origin, NativeMtpDraftOrigin::InitialSerial);
        let decision = verifier.observe_taken_draft_verification(pending.token, 13, 9);

        assert_eq!(
            decision,
            NativeMtpVerification::Rejected {
                draft: 12,
                target: 13,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 1,
                rejected_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 7,
                verification_compute_us: 9,
                ..NativeMtpN1Stats::default()
            }
        );
    }

    #[test]
    fn clear_pending_draft_drops_unverified_draft_without_changing_stats() {
        let mut verifier = NativeMtpN1Verifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        verifier.clear_pending_draft();

        assert_eq!(
            verifier.stats(),
            NativeMtpN1Stats {
                drafted_tokens: 1,
                proposal_compute_us: 7,
                ..NativeMtpN1Stats::default()
            }
        );
        assert_eq!(
            observe(&mut verifier, 12, 9, None),
            NativeMtpVerification::NoPending
        );
    }

    #[test]
    fn verification_compute_time_saturates_instead_of_overflowing() {
        let mut verifier = NativeMtpN1Verifier::default();

        observe(&mut verifier, 11, i64::MAX, Some(draft(12)));
        observe(&mut verifier, 13, i64::MAX, Some(draft(14)));
        observe(&mut verifier, 15, 1, None);

        assert_eq!(verifier.stats().verification_compute_us, i64::MAX);
    }
}
