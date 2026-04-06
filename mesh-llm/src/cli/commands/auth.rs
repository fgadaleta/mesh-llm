use std::path::PathBuf;

use anyhow::{bail, Result};
use zeroize::Zeroizing;

use crate::crypto::{
    default_keystore_path, keystore_exists, keystore_metadata, load_keystore,
    load_owner_keypair_from_keychain, save_keystore, save_keystore_with_keychain,
    OwnerKeychainLoadError, OwnerKeypair, KEYCHAIN_SERVICE,
};

/// Run `mesh-llm auth init`.
pub(crate) fn run_init(
    owner_key: Option<PathBuf>,
    force: bool,
    no_passphrase: bool,
    keychain: bool,
) -> Result<()> {
    let path = match owner_key {
        Some(p) => p,
        None => default_keystore_path()?,
    };

    let existing_keystore = keystore_exists(&path);
    if existing_keystore && !force {
        bail!(
            "Owner keystore already exists at {}\nUse --force to overwrite.",
            path.display()
        );
    }

    let use_keychain = if keychain {
        if !crate::crypto::keychain_available() {
            bail!(
                "No OS keychain backend is available on this host.\n\
                 Retry without --keychain to set a passphrase, or with --no-passphrase \
                 to store keys unencrypted."
            );
        }
        true
    } else {
        // Only probe the OS keychain when it could possibly be used: new
        // keystore and passphrase flow. Probing unconditionally would trigger
        // D-Bus round-trips on Linux even when --no-passphrase is set or when
        // we are overwriting an existing keystore (neither of which can
        // default to keychain).
        let available =
            (!existing_keystore && !no_passphrase) && crate::crypto::keychain_available();
        should_default_to_keychain(existing_keystore, no_passphrase, available)
    };

    let keypair = OwnerKeypair::generate();
    let owner_id = keypair.owner_id();
    let sign_pk = hex::encode(keypair.verifying_key().as_bytes());
    let enc_pk = hex::encode(keypair.encryption_public_key().as_bytes());
    let source = if no_passphrase {
        save_keystore(&path, &keypair, None, force)?;
        PassphraseSource::None
    } else if use_keychain {
        let account = save_keystore_with_keychain(&path, &keypair, force)?;
        PassphraseSource::Keychain { account }
    } else {
        let pass = Zeroizing::new(rpassword::prompt_password_stderr(
            "Enter passphrase (empty for none): ",
        )?);
        if pass.is_empty() {
            save_keystore(&path, &keypair, None, force)?;
            PassphraseSource::None
        } else {
            let confirm =
                Zeroizing::new(rpassword::prompt_password_stderr("Confirm passphrase: ")?);
            if pass.as_str() != confirm.as_str() {
                bail!("Passphrases do not match.");
            }
            save_keystore(&path, &keypair, Some(pass.as_str()), force)?;
            PassphraseSource::Prompt
        }
    };
    let encrypted = !matches!(source, PassphraseSource::None);

    eprintln!();
    eprintln!("Owner keystore created.");
    eprintln!("Owner ID:        {owner_id}");
    eprintln!("Signing key:     {sign_pk}");
    eprintln!("Encryption key:  {enc_pk}");
    eprintln!("Path:            {}", path.display());
    eprintln!("Encrypted:       {}", if encrypted { "yes" } else { "no" });
    match &source {
        PassphraseSource::Keychain { account, .. } => {
            eprintln!(
                "Unlock:          OS keychain (service={KEYCHAIN_SERVICE}, account={account})"
            );
        }
        PassphraseSource::Prompt => {
            eprintln!("Unlock:          passphrase prompt");
        }
        PassphraseSource::None => {}
    }
    eprintln!();
    eprintln!("Next steps:");
    match &source {
        PassphraseSource::Keychain { account, .. } => {
            eprintln!(
                "- This keystore is unlock-bound to this machine's keychain. To share the same \
                 owner identity on another node, retrieve the passphrase from your OS keychain \
                 (service={KEYCHAIN_SERVICE}, account={account}) and enter it there, or re-run \
                 `auth init` with a manual passphrase so the same passphrase can be used \
                 everywhere."
            );
        }
        PassphraseSource::Prompt | PassphraseSource::None => {
            eprintln!(
                "- Copy this keystore to other trusted nodes that should share the same owner identity."
            );
        }
    }
    eprintln!("- Start mesh-llm with --owner-key {}", path.display());

    Ok(())
}

/// Run `mesh-llm auth status`.
pub(crate) fn run_status(owner_key: Option<PathBuf>) -> Result<()> {
    let path = match owner_key {
        Some(p) => p,
        None => default_keystore_path()?,
    };

    if !keystore_exists(&path) {
        eprintln!("No owner keystore found at {}", path.display());
        eprintln!("Run `mesh-llm auth init` to create one.");
        return Ok(());
    }

    let info = keystore_metadata(&path)?;

    eprintln!("Owner keystore:  {}", path.display());
    eprintln!("Status:          present");
    eprintln!(
        "Encrypted:       {}",
        if info.encrypted { "yes" } else { "no" }
    );
    eprintln!("Owner ID:        {}", info.owner_id);
    if let Some(ref spk) = info.signing_public_key {
        eprintln!("Signing key:     {spk}");
    }
    if let Some(ref epk) = info.encryption_public_key {
        eprintln!("Encryption key:  {epk}");
    }
    eprintln!("Created:         {}", info.created_at);

    if info.encrypted {
        match load_owner_keypair_from_keychain(&path) {
            Ok(_) => {
                eprintln!("Keystore:        valid (unlocked from OS keychain)");
            }
            Err(OwnerKeychainLoadError::Crypto(e)) => {
                eprintln!(
                    "{}",
                    encrypted_keystore_keychain_status(OwnerKeychainLoadError::Crypto(e))
                );
            }
            Err(e) => eprintln!("{}", encrypted_keystore_keychain_status(e)),
        }
    } else {
        match load_keystore(&path, None) {
            Ok(_) => {
                eprintln!("Keystore:        valid (keys loaded successfully)");
            }
            Err(e) => {
                eprintln!("Keystore:        ERROR loading keys: {e}");
            }
        }
    }

    Ok(())
}

// ── Internal helpers ────────────────────────────────────────────────

#[derive(Debug)]
enum PassphraseSource {
    None,
    Prompt,
    Keychain { account: String },
}

fn encrypted_keystore_keychain_status(error: OwnerKeychainLoadError) -> String {
    match error {
        OwnerKeychainLoadError::NoEntry => {
            "Keystore:        encrypted (no keychain entry for this path; provide the \
             passphrase when the owner keystore is consumed)"
                .into()
        }
        OwnerKeychainLoadError::Crypto(crate::crypto::CryptoError::DecryptionFailed) => {
            "Keystore:        encrypted (keychain entry could not unlock this keystore; \
             provide the passphrase when the owner keystore is consumed or remove the stale \
             keychain entry for this path)"
                .into()
        }
        OwnerKeychainLoadError::Crypto(crate::crypto::CryptoError::KeychainUnavailable {
            reason,
        }) => format!("Keystore:        encrypted (keychain unavailable: {reason})"),
        OwnerKeychainLoadError::Crypto(crate::crypto::CryptoError::KeychainAccessDenied {
            reason,
        }) => format!(
            "Keystore:        encrypted (keychain is locked or access was denied: {reason}; \
             unlock the keychain and retry, or provide the passphrase when the owner keystore \
             is consumed)"
        ),
        OwnerKeychainLoadError::Crypto(e) => format!("Keystore:        ERROR loading keys: {e}"),
    }
}

fn should_default_to_keychain(
    existing_keystore: bool,
    no_passphrase: bool,
    keychain_available: bool,
) -> bool {
    !existing_keystore && !no_passphrase && keychain_available
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn defaults_to_keychain_for_new_keystore_when_available() {
        assert!(should_default_to_keychain(false, false, true));
    }

    #[test]
    fn does_not_default_to_keychain_for_existing_keystore() {
        assert!(!should_default_to_keychain(true, false, true));
    }

    #[test]
    fn does_not_default_to_keychain_when_unavailable() {
        assert!(!should_default_to_keychain(false, false, false));
    }

    #[test]
    fn does_not_default_to_keychain_with_no_passphrase() {
        assert!(!should_default_to_keychain(false, true, true));
    }

    #[test]
    fn reports_stale_keychain_entry_as_encrypted_keystore() {
        let message = encrypted_keystore_keychain_status(OwnerKeychainLoadError::Crypto(
            crate::crypto::CryptoError::DecryptionFailed,
        ));

        assert!(message.contains("keychain entry could not unlock this keystore"));
        assert!(message.contains("remove the stale keychain entry for this path"));
    }

    #[test]
    #[serial]
    fn force_keychain_save_failure_restores_previous_secret() {
        if !crate::crypto::keychain_available() {
            eprintln!("keychain backend unavailable, skipping");
            return;
        }

        // Use a path whose parent is a FILE rather than a directory. That
        // makes the save_keystore call fail at create_dir_all, triggering
        // our rollback logic.
        let tmp_dir =
            std::env::temp_dir().join(format!("mesh-llm-force-rollback-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let blocking_file = tmp_dir.join("blocker");
        std::fs::write(&blocking_file, b"not a directory").unwrap();
        // This path uses `blocker` (a regular file) as if it were a parent
        // directory, so any attempt to create_dir_all against it fails.
        let bad_path = blocking_file.join("owner-keystore.json");

        // Seed the keychain with a "previous" unlock secret at the account
        // that run_init will compute for bad_path.
        let account = crate::crypto::owner_keychain_account_for_path(&bad_path);
        let previous_secret = "previous-unlock-secret-do-not-lose";
        crate::crypto::keychain_set(KEYCHAIN_SERVICE, &account, previous_secret).unwrap();

        // Force-init with keychain. Expected to fail because the parent of
        // bad_path is a regular file. Our rollback should restore the
        // previous secret we seeded.
        let result = run_init(Some(bad_path.clone()), true, false, true);
        assert!(
            result.is_err(),
            "run_init must fail when save cannot succeed"
        );

        let restored = crate::crypto::keychain_get(KEYCHAIN_SERVICE, &account).unwrap();
        assert_eq!(
            restored.as_deref(),
            Some(previous_secret),
            "previous keychain secret must be restored after failed force-init"
        );

        // Cleanup.
        crate::crypto::keychain_delete(KEYCHAIN_SERVICE, &account).ok();
        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    #[test]
    #[serial]
    fn fresh_keychain_save_failure_leaves_no_orphan() {
        if !crate::crypto::keychain_available() {
            eprintln!("keychain backend unavailable, skipping");
            return;
        }

        let tmp_dir =
            std::env::temp_dir().join(format!("mesh-llm-fresh-rollback-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let blocking_file = tmp_dir.join("blocker");
        std::fs::write(&blocking_file, b"not a directory").unwrap();
        let bad_path = blocking_file.join("owner-keystore.json");

        let account = crate::crypto::owner_keychain_account_for_path(&bad_path);

        // Make sure no entry exists at that account before we start.
        crate::crypto::keychain_delete(KEYCHAIN_SERVICE, &account).ok();

        let result = run_init(Some(bad_path.clone()), false, false, true);
        assert!(
            result.is_err(),
            "run_init must fail when save cannot succeed"
        );

        let residual = crate::crypto::keychain_get(KEYCHAIN_SERVICE, &account).unwrap();
        assert_eq!(
            residual, None,
            "a fresh init failure must leave no keychain entry behind"
        );

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    #[test]
    #[serial]
    fn init_defaults_to_keychain_then_load_round_trip() {
        if !crate::crypto::keychain_available() {
            eprintln!("keychain backend unavailable, skipping");
            return;
        }

        // Use a unique temp path per run so tests can be repeated / run in
        // parallel without stomping each other.
        let dir =
            std::env::temp_dir().join(format!("mesh-llm-keychain-rt-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("owner-keystore.json");

        run_init(Some(path.clone()), false, false, false)
            .expect("auth init should default to keychain when available");

        // Keystore file exists and is encrypted.
        assert!(path.exists(), "keystore file should exist");
        let info = keystore_metadata(&path).unwrap();
        assert!(
            info.encrypted,
            "keystore should be encrypted when using keychain"
        );

        // Keychain has an entry for this path's account.
        let account = crate::crypto::owner_keychain_account_for_path(&path);
        let stored = crate::crypto::keychain_get(KEYCHAIN_SERVICE, &account).unwrap();
        assert!(
            stored.is_some(),
            "keychain must have a passphrase entry for this keystore path"
        );

        // Loading through the keychain helper succeeds and yields the same
        // owner id the keystore metadata advertises.
        let kp = load_owner_keypair_from_keychain(&path).expect("load via keychain must succeed");
        assert_eq!(kp.owner_id(), info.owner_id);

        // Cleanup: remove keychain entry + temp dir.
        crate::crypto::keychain_delete(KEYCHAIN_SERVICE, &account).ok();
        std::fs::remove_dir_all(&dir).ok();
    }
}
