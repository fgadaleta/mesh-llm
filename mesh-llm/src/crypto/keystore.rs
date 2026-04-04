use std::io::Write;
use std::path::{Path, PathBuf};

use argon2::Argon2;
use chacha20poly1305::aead::{Aead, AeadCore, OsRng};
use chacha20poly1305::{ChaCha20Poly1305, KeyInit};
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use super::error::CryptoError;
use super::keys::OwnerKeypair;

const KEYSTORE_VERSION: u32 = 1;

/// On-disk keystore format (JSON).
#[derive(Serialize, Deserialize)]
struct KeystoreV1 {
    version: u32,
    owner_id: String,
    created_at: String,
    encrypted: bool,

    // Present when encrypted == false
    signing_secret_key: Option<String>,
    signing_public_key: Option<String>,
    encryption_secret_key: Option<String>,
    encryption_public_key: Option<String>,

    // Present when encrypted == true
    kdf: Option<String>,
    argon2_salt: Option<String>,
    nonce: Option<String>,
    ciphertext: Option<String>,
}

/// Plaintext inner payload that gets encrypted when passphrase is set.
#[derive(Serialize, Deserialize)]
struct SecretPayload {
    signing_secret_key: String,
    encryption_secret_key: String,
}

/// Return the default keystore path: `~/.mesh-llm/owner-keystore.json`.
pub fn default_keystore_path() -> Result<PathBuf, CryptoError> {
    let home = dirs::home_dir().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot determine home directory",
        ))
    })?;
    Ok(home.join(".mesh-llm").join("owner-keystore.json"))
}

/// Check if a keystore file exists at the given path.
pub fn keystore_exists(path: &Path) -> bool {
    path.exists()
}

/// Save an owner keypair to disk.
///
/// If `passphrase` is `Some`, the secret keys are encrypted with Argon2id + ChaCha20Poly1305.
/// File permissions are set to 0600 on Unix.
pub fn save_keystore(
    path: &Path,
    keypair: &OwnerKeypair,
    passphrase: Option<&str>,
) -> Result<(), CryptoError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let keystore = if let Some(pass) = passphrase {
        build_encrypted_keystore(keypair, pass)?
    } else {
        build_plaintext_keystore(keypair)
    };

    let json = serde_json::to_string_pretty(&keystore)?;
    write_keystore_atomically(path, json.as_bytes())?;
    Ok(())
}

/// Load an owner keypair from disk.
///
/// If the keystore is encrypted, `passphrase` must be provided.
pub fn load_keystore(path: &Path, passphrase: Option<&str>) -> Result<OwnerKeypair, CryptoError> {
    if !path.exists() {
        return Err(CryptoError::KeystoreNotFound {
            path: path.display().to_string(),
        });
    }

    let raw = std::fs::read_to_string(path)?;
    let ks: KeystoreV1 = serde_json::from_str(&raw)?;

    if ks.version != KEYSTORE_VERSION {
        return Err(CryptoError::UnsupportedVersion {
            version: ks.version,
        });
    }

    if ks.encrypted {
        decrypt_keystore(&ks, passphrase)
    } else {
        plaintext_keystore(&ks)
    }
}

/// Read keystore metadata without decrypting secret keys.
pub fn keystore_metadata(path: &Path) -> Result<KeystoreInfo, CryptoError> {
    if !path.exists() {
        return Err(CryptoError::KeystoreNotFound {
            path: path.display().to_string(),
        });
    }
    let raw = std::fs::read_to_string(path)?;
    let ks: KeystoreV1 = serde_json::from_str(&raw)?;
    let signing_public_key =
        ks.signing_public_key
            .clone()
            .ok_or_else(|| CryptoError::InvalidKeyMaterial {
                reason: "missing signing_public_key in keystore metadata".into(),
            })?;
    let signing_public_key_bytes: [u8; 32] = hex::decode(&signing_public_key)
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("bad signing public key hex: {e}"),
        })?
        .try_into()
        .map_err(|_| CryptoError::InvalidKeyMaterial {
            reason: "signing public key must be 32 bytes".into(),
        })?;
    let signing_public_key = ed25519_dalek::VerifyingKey::from_bytes(&signing_public_key_bytes)
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("invalid signing public key: {e}"),
        })?;
    let verified_owner_id = super::keys::owner_id_from_verifying_key(&signing_public_key);
    if ks.owner_id != verified_owner_id {
        return Err(CryptoError::VerificationFailed {
            reason: "owner_id does not match signing public key".into(),
        });
    }

    Ok(KeystoreInfo {
        owner_id: verified_owner_id,
        created_at: ks.created_at,
        encrypted: ks.encrypted,
        signing_public_key: Some(hex::encode(signing_public_key.as_bytes())),
        encryption_public_key: ks.encryption_public_key,
    })
}

/// Public metadata about a keystore (no secrets).
pub struct KeystoreInfo {
    pub owner_id: String,
    pub created_at: String,
    pub encrypted: bool,
    pub signing_public_key: Option<String>,
    pub encryption_public_key: Option<String>,
}

// ── Internal helpers ────────────────────────────────────────────────

fn build_plaintext_keystore(keypair: &OwnerKeypair) -> KeystoreV1 {
    KeystoreV1 {
        version: KEYSTORE_VERSION,
        owner_id: keypair.owner_id(),
        created_at: chrono::Utc::now().to_rfc3339(),
        encrypted: false,
        signing_secret_key: Some(hex::encode(keypair.signing_bytes())),
        signing_public_key: Some(hex::encode(keypair.verifying_key().as_bytes())),
        encryption_secret_key: Some(hex::encode(keypair.encryption_bytes())),
        encryption_public_key: Some(hex::encode(keypair.encryption_public_key().as_bytes())),
        kdf: None,
        argon2_salt: None,
        nonce: None,
        ciphertext: None,
    }
}

fn build_encrypted_keystore(
    keypair: &OwnerKeypair,
    passphrase: &str,
) -> Result<KeystoreV1, CryptoError> {
    let salt: [u8; 16] = rand::random();

    // Derive a 32-byte symmetric key from the passphrase.
    let sym_key = derive_key(passphrase, &salt)?;

    // Encrypt the secret keys.
    let payload = SecretPayload {
        signing_secret_key: hex::encode(keypair.signing_bytes()),
        encryption_secret_key: hex::encode(keypair.encryption_bytes()),
    };
    let plaintext = serde_json::to_vec(&payload)?;

    let cipher = ChaCha20Poly1305::new(chacha20poly1305::Key::from_slice(sym_key.as_ref()));
    let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
    let ct = cipher
        .encrypt(&nonce, plaintext.as_ref())
        .map_err(|_| CryptoError::DecryptionFailed)?;

    Ok(KeystoreV1 {
        version: KEYSTORE_VERSION,
        owner_id: keypair.owner_id(),
        created_at: chrono::Utc::now().to_rfc3339(),
        encrypted: true,
        signing_secret_key: None,
        signing_public_key: Some(hex::encode(keypair.verifying_key().as_bytes())),
        encryption_secret_key: None,
        encryption_public_key: Some(hex::encode(keypair.encryption_public_key().as_bytes())),
        kdf: Some("argon2id-chacha20poly1305".into()),
        argon2_salt: Some(hex::encode(salt)),
        nonce: Some(hex::encode(nonce)),
        ciphertext: Some(hex::encode(ct)),
    })
}

fn write_keystore_atomically(path: &Path, bytes: &[u8]) -> Result<(), CryptoError> {
    let parent = path.parent().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("keystore path {} has no parent directory", path.display()),
        ))
    })?;
    let file_name = path.file_name().ok_or_else(|| {
        CryptoError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("keystore path {} has no file name", path.display()),
        ))
    })?;
    let tmp_path = parent.join(format!(
        ".{}.tmp-{}-{}",
        file_name.to_string_lossy(),
        std::process::id(),
        rand::random::<u64>()
    ));

    let write_result = (|| -> Result<(), CryptoError> {
        let mut options = std::fs::OpenOptions::new();
        options.create_new(true).write(true);

        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;

            options.mode(0o600);
        }

        let mut file = options.open(&tmp_path)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);

        std::fs::rename(&tmp_path, path)?;
        Ok(())
    })();

    if write_result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }

    write_result
}

fn plaintext_keystore(ks: &KeystoreV1) -> Result<OwnerKeypair, CryptoError> {
    let sign_hex = ks
        .signing_secret_key
        .as_deref()
        .ok_or(CryptoError::InvalidKeyMaterial {
            reason: "missing signing_secret_key in plaintext keystore".into(),
        })?;
    let enc_hex = ks
        .encryption_secret_key
        .as_deref()
        .ok_or(CryptoError::InvalidKeyMaterial {
            reason: "missing encryption_secret_key in plaintext keystore".into(),
        })?;

    let sign_bytes = hex::decode(sign_hex).map_err(|e| CryptoError::InvalidKeyMaterial {
        reason: format!("bad signing key hex: {e}"),
    })?;
    let enc_bytes = hex::decode(enc_hex).map_err(|e| CryptoError::InvalidKeyMaterial {
        reason: format!("bad encryption key hex: {e}"),
    })?;

    OwnerKeypair::from_bytes(&sign_bytes, &enc_bytes)
}

fn decrypt_keystore(
    ks: &KeystoreV1,
    passphrase: Option<&str>,
) -> Result<OwnerKeypair, CryptoError> {
    let pass = passphrase.ok_or(CryptoError::DecryptionFailed)?;

    let salt = hex::decode(
        ks.argon2_salt
            .as_deref()
            .ok_or(CryptoError::DecryptionFailed)?,
    )
    .map_err(|_| CryptoError::DecryptionFailed)?;
    let nonce_bytes = hex::decode(ks.nonce.as_deref().ok_or(CryptoError::DecryptionFailed)?)
        .map_err(|_| CryptoError::DecryptionFailed)?;
    let ct = hex::decode(
        ks.ciphertext
            .as_deref()
            .ok_or(CryptoError::DecryptionFailed)?,
    )
    .map_err(|_| CryptoError::DecryptionFailed)?;

    if salt.len() != 16 || nonce_bytes.len() != 12 || ct.is_empty() {
        return Err(CryptoError::DecryptionFailed);
    }

    let sym_key = derive_key(pass, &salt)?;
    let cipher = ChaCha20Poly1305::new(chacha20poly1305::Key::from_slice(sym_key.as_ref()));
    let nonce = chacha20poly1305::Nonce::from_slice(&nonce_bytes);
    let plaintext = cipher
        .decrypt(nonce, ct.as_ref())
        .map_err(|_| CryptoError::DecryptionFailed)?;

    let payload: SecretPayload =
        serde_json::from_slice(&plaintext).map_err(|_| CryptoError::DecryptionFailed)?;

    let sign_bytes =
        hex::decode(&payload.signing_secret_key).map_err(|_| CryptoError::DecryptionFailed)?;
    let enc_bytes =
        hex::decode(&payload.encryption_secret_key).map_err(|_| CryptoError::DecryptionFailed)?;

    OwnerKeypair::from_bytes(&sign_bytes, &enc_bytes)
}

fn derive_key(passphrase: &str, salt: &[u8]) -> Result<Zeroizing<[u8; 32]>, CryptoError> {
    let mut key = Zeroizing::new([0u8; 32]);
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), salt, key.as_mut())
        .map_err(|e| CryptoError::InvalidKeyMaterial {
            reason: format!("argon2 KDF failed: {e}"),
        })?;
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_keystore_path() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mesh-llm-test-{}", rand::random::<u64>()));
        fs::create_dir_all(&dir).unwrap();
        dir.join("owner-keystore.json")
    }

    #[test]
    fn round_trip_plaintext() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();
        let original_id = kp.owner_id();

        save_keystore(&path, &kp, None).unwrap();
        let loaded = load_keystore(&path, None).unwrap();

        assert_eq!(original_id, loaded.owner_id());
        assert_eq!(kp.signing_bytes(), loaded.signing_bytes());
        assert_eq!(kp.encryption_bytes(), loaded.encryption_bytes());

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn round_trip_encrypted() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();
        let original_id = kp.owner_id();

        save_keystore(&path, &kp, Some("test-passphrase")).unwrap();
        let loaded = load_keystore(&path, Some("test-passphrase")).unwrap();

        assert_eq!(original_id, loaded.owner_id());
        assert_eq!(kp.signing_bytes(), loaded.signing_bytes());
        assert_eq!(kp.encryption_bytes(), loaded.encryption_bytes());

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn wrong_passphrase_fails() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();

        save_keystore(&path, &kp, Some("correct")).unwrap();

        let result = load_keystore(&path, Some("wrong"));
        assert!(
            matches!(result, Err(CryptoError::DecryptionFailed)),
            "expected DecryptionFailed, got {result:?}"
        );

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn missing_passphrase_for_encrypted_fails() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();

        save_keystore(&path, &kp, Some("secret")).unwrap();

        let result = load_keystore(&path, None);
        assert!(
            matches!(result, Err(CryptoError::DecryptionFailed)),
            "expected DecryptionFailed, got {result:?}"
        );

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn keystore_not_found() {
        let path = PathBuf::from("/tmp/nonexistent-keystore.json");
        let result = load_keystore(&path, None);
        assert!(matches!(result, Err(CryptoError::KeystoreNotFound { .. })));
    }

    #[test]
    fn metadata_without_decryption() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();
        let expected_id = kp.owner_id();

        save_keystore(&path, &kp, Some("secret")).unwrap();

        let info = keystore_metadata(&path).unwrap();
        assert_eq!(info.owner_id, expected_id);
        assert!(info.encrypted);

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn metadata_rejects_owner_id_spoofing() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();

        save_keystore(&path, &kp, Some("secret")).unwrap();

        let mut raw: KeystoreV1 =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        raw.owner_id = "0000000000000000000000000000000000000000000000000000000000000000".into();
        fs::write(&path, serde_json::to_string_pretty(&raw).unwrap()).unwrap();

        let result = keystore_metadata(&path);
        assert!(
            matches!(result, Err(CryptoError::VerificationFailed { .. })),
            "spoofed owner_id should be rejected"
        );

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn encrypted_keystore_requires_all_encrypted_fields() {
        let path = temp_keystore_path();
        let kp = OwnerKeypair::generate();

        save_keystore(&path, &kp, Some("secret")).unwrap();

        let mut raw: KeystoreV1 =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        raw.argon2_salt = None;
        fs::write(&path, serde_json::to_string_pretty(&raw).unwrap()).unwrap();

        let result = load_keystore(&path, Some("secret"));
        assert!(
            matches!(result, Err(CryptoError::DecryptionFailed)),
            "missing encrypted fields should fail cleanly"
        );

        fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
