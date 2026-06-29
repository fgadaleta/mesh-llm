use std::io::{self, Read, Write};

use super::{
    MAX_STAGE_ACTIVATION_BYTES, MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES,
    MAX_STAGE_DECODED_ACTIVATION_BYTES, MAX_STAGE_LOGIT_BIAS, MAX_STAGE_PREDICTED_TOKENS,
    MAX_STAGE_SIDEBAND_VALUES, MAX_STAGE_STATE_IMPORT_BYTES, READY_MAGIC, STAGE_STATE_VERSION,
    StageLogitBias, StageReply, StageReplyStats, StageSamplingConfig, StageStateHeader,
    StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind,
    activation::{
        activation_decoded_f32_bytes_with_state_flags, activation_wire_bytes_with_state_flags,
    },
    invalid_data, invalid_input,
};

pub fn send_ready(mut writer: impl Write) -> io::Result<()> {
    write_i32(&mut writer, READY_MAGIC)
}

pub fn recv_ready(mut reader: impl Read) -> io::Result<()> {
    let magic = read_i32(&mut reader)?;
    if magic != READY_MAGIC {
        return Err(invalid_data("stage ready magic mismatch"));
    }
    Ok(())
}

pub fn send_reply_ack(mut writer: impl Write) -> io::Result<()> {
    send_reply_ack_with_stats(&mut writer, StageReplyStats::default())
}

pub fn send_reply_ack_with_stats(mut writer: impl Write, stats: StageReplyStats) -> io::Result<()> {
    write_i32(&mut writer, WireReplyKind::Ack as i32)?;
    write_i32(&mut writer, 0)?;
    write_i32(&mut writer, 0)?;
    write_reply_stats(&mut writer, stats)
}

pub fn send_reply_predicted(mut writer: impl Write, predicted: i32) -> io::Result<()> {
    send_reply_predicted_with_stats(&mut writer, predicted, StageReplyStats::default())
}

pub fn send_reply_predicted_with_stats(
    mut writer: impl Write,
    predicted: i32,
    stats: StageReplyStats,
) -> io::Result<()> {
    send_reply_predicted_with_tokens_and_stats(&mut writer, predicted, &[predicted], stats)
}

pub fn send_reply_predicted_with_tokens_and_stats(
    mut writer: impl Write,
    predicted: i32,
    predicted_tokens: &[i32],
    stats: StageReplyStats,
) -> io::Result<()> {
    if predicted_tokens.len() > MAX_STAGE_PREDICTED_TOKENS {
        return Err(invalid_input("too many predicted tokens"));
    }
    write_i32(&mut writer, WireReplyKind::PredictedToken as i32)?;
    write_i32(&mut writer, predicted)?;
    write_i32(
        &mut writer,
        i32::try_from(predicted_tokens.len())
            .map_err(|_| invalid_input("too many predicted tokens"))?,
    )?;
    for token in predicted_tokens {
        write_i32(&mut writer, *token)?;
    }
    write_reply_stats(&mut writer, stats)
}

pub fn send_reply_predicted_tokens_with_stats(
    mut writer: impl Write,
    predicted_tokens: &[i32],
    stats: StageReplyStats,
) -> io::Result<()> {
    if predicted_tokens.len() > MAX_STAGE_PREDICTED_TOKENS {
        return Err(invalid_input("too many predicted tokens"));
    }
    let predicted = predicted_tokens.first().copied().unwrap_or(0);
    write_i32(&mut writer, WireReplyKind::PredictedTokens as i32)?;
    write_i32(&mut writer, predicted)?;
    write_i32(
        &mut writer,
        i32::try_from(predicted_tokens.len())
            .map_err(|_| invalid_input("too many predicted tokens"))?,
    )?;
    for token in predicted_tokens {
        write_i32(&mut writer, *token)?;
    }
    write_reply_stats(&mut writer, stats)
}

pub fn recv_reply(mut reader: impl Read) -> io::Result<StageReply> {
    let kind = WireReplyKind::try_from(read_i32(&mut reader)?)?;
    let predicted = read_i32(&mut reader)?;
    let predicted_count = checked_i32_len(
        read_i32(&mut reader)?,
        MAX_STAGE_PREDICTED_TOKENS,
        "negative predicted token count",
        "predicted token count exceeds maximum",
    )?;
    let mut predicted_tokens = Vec::with_capacity(predicted_count);
    for _ in 0..predicted_count {
        predicted_tokens.push(read_i32(&mut reader)?);
    }
    let stats = read_reply_stats(&mut reader)?;
    Ok(StageReply {
        kind,
        predicted,
        predicted_tokens,
        stats,
    })
}

pub fn write_stage_message(
    mut writer: impl Write,
    message: &StageWireMessage,
    dtype: WireActivationDType,
) -> io::Result<()> {
    // Wire v4 fixed prefix, little-endian:
    // kind, pos_start, token_count, token_sideband_count, position_sideband_count (5 x i32);
    // StageStateHeader (10 x i32); request_id, session_id (2 x u64);
    // optional StageSamplingConfig follows when state_flags::SAMPLING is set.
    // Token sideband, raw StateImport bytes, or activation bytes follow this
    // prefix, so prefill overhead stays independent of ID string length.
    write_i32(&mut writer, message.kind as i32)?;
    write_i32(&mut writer, message.pos_start)?;
    write_i32(&mut writer, message.token_count)?;
    if message.tokens.len() > MAX_STAGE_SIDEBAND_VALUES {
        return Err(invalid_input("too many tokens"));
    }
    write_i32(
        &mut writer,
        i32::try_from(message.tokens.len()).map_err(|_| invalid_input("too many tokens"))?,
    )?;
    if message.positions.len() > MAX_STAGE_SIDEBAND_VALUES {
        return Err(invalid_input("too many position sideband values"));
    }
    write_i32(
        &mut writer,
        i32::try_from(message.positions.len())
            .map_err(|_| invalid_input("too many position sideband values"))?,
    )?;

    let mut state = message.state;
    state.reserved = dtype as i32;
    if message.sampling.is_some() {
        state.flags |= super::state_flags::SAMPLING;
    } else {
        state.flags &= !super::state_flags::SAMPLING;
    }
    if message.chat_sampling_metadata.is_some() {
        state.flags |= super::state_flags::CHAT_SAMPLING_METADATA;
    } else {
        state.flags &= !super::state_flags::CHAT_SAMPLING_METADATA;
    }
    write_state_header(&mut writer, state)?;
    write_u64(&mut writer, message.request_id)?;
    write_u64(&mut writer, message.session_id)?;
    if let Some(sampling) = message.sampling.as_ref() {
        write_sampling_config(&mut writer, sampling)?;
    }
    if let Some(metadata) = message.chat_sampling_metadata.as_ref() {
        let bytes = metadata.as_bytes();
        if bytes.len() > MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES {
            return Err(invalid_input("chat sampling metadata is too large"));
        }
        write_u32(
            &mut writer,
            u32::try_from(bytes.len())
                .map_err(|_| invalid_input("chat sampling metadata is too large"))?,
        )?;
        writer.write_all(bytes)?;
    }

    if message.kind == WireMessageKind::StateImport {
        let raw_byte_count = usize::try_from(message.token_count)
            .map_err(|_| invalid_input("state import raw byte count mismatch"))?;
        if raw_byte_count != message.raw_bytes.len() {
            return Err(invalid_input("state import raw byte count mismatch"));
        }
        if raw_byte_count > MAX_STAGE_STATE_IMPORT_BYTES {
            return Err(invalid_input("state import raw byte count exceeds maximum"));
        }
        writer.write_all(&message.raw_bytes)?;
        return Ok(());
    }
    for token in &message.tokens {
        write_i32(&mut writer, *token)?;
    }
    for position in &message.positions {
        write_i32(&mut writer, *position)?;
    }
    writer.write_all(&message.activation)?;
    Ok(())
}

pub fn read_stage_message(mut reader: impl Read, n_embd: i32) -> io::Result<StageWireMessage> {
    let kind = WireMessageKind::try_from(read_i32(&mut reader)?)?;
    let pos_start = read_i32(&mut reader)?;
    let token_count = read_i32(&mut reader)?;
    let token_sideband_count = read_i32(&mut reader)?;
    let position_sideband_count = read_i32(&mut reader)?;
    let state = read_state_header(&mut reader)?;
    if state.version != STAGE_STATE_VERSION {
        return Err(invalid_data("unsupported stage state version"));
    }
    let request_id = read_u64(&mut reader)?;
    let session_id = read_u64(&mut reader)?;
    let sampling = if (state.flags & super::state_flags::SAMPLING) != 0 {
        Some(read_sampling_config(&mut reader)?)
    } else {
        None
    };
    let chat_sampling_metadata = if (state.flags & super::state_flags::CHAT_SAMPLING_METADATA) != 0
    {
        let len = checked_u32_len(
            read_u32(&mut reader)?,
            MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES,
            "chat sampling metadata length exceeds maximum",
        )?;
        let mut bytes = vec![0_u8; len];
        reader.read_exact(&mut bytes)?;
        Some(
            String::from_utf8(bytes)
                .map_err(|_| invalid_data("chat sampling metadata is not UTF-8"))?,
        )
    } else {
        None
    };
    let dtype = state.dtype()?;
    if kind == WireMessageKind::Stop {
        return Ok(StageWireMessage {
            kind,
            pos_start,
            token_count,
            state,
            request_id,
            session_id,
            sampling,
            chat_sampling_metadata,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        });
    }
    if token_count < 0 || token_sideband_count < 0 || position_sideband_count < 0 {
        return Err(invalid_data("negative wire count"));
    }
    let token_sideband_count = checked_i32_len(
        token_sideband_count,
        MAX_STAGE_SIDEBAND_VALUES,
        "negative wire count",
        "token sideband count exceeds maximum",
    )?;
    let position_sideband_count = checked_i32_len(
        position_sideband_count,
        MAX_STAGE_SIDEBAND_VALUES,
        "negative wire count",
        "position sideband count exceeds maximum",
    )?;
    if kind == WireMessageKind::StateImport {
        let raw_byte_count = checked_i32_len(
            token_count,
            MAX_STAGE_STATE_IMPORT_BYTES,
            "negative wire count",
            "state import byte count exceeds maximum",
        )?;
        let mut raw_bytes = vec![0; raw_byte_count];
        reader.read_exact(&mut raw_bytes)?;
        return Ok(StageWireMessage {
            kind,
            pos_start,
            token_count,
            state,
            request_id,
            session_id,
            sampling,
            chat_sampling_metadata,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes,
        });
    }

    let mut tokens = Vec::with_capacity(token_sideband_count);
    for _ in 0..token_sideband_count {
        tokens.push(read_i32(&mut reader)?);
    }
    let mut positions = Vec::with_capacity(position_sideband_count);
    for _ in 0..position_sideband_count {
        positions.push(read_i32(&mut reader)?);
    }
    let activation_bytes =
        if state.source_stage_index < 0 || kind.is_activationless_prefix_cache_control() {
            0
        } else {
            activation_wire_bytes_with_state_flags(dtype, token_count, n_embd, state.flags)?
        };
    if activation_bytes > MAX_STAGE_ACTIVATION_BYTES {
        return Err(invalid_data(
            "activation payload byte count exceeds maximum",
        ));
    }
    if activation_bytes > 0 {
        let decoded_activation_bytes =
            activation_decoded_f32_bytes_with_state_flags(token_count, n_embd, state.flags)?;
        if decoded_activation_bytes > MAX_STAGE_DECODED_ACTIVATION_BYTES {
            return Err(invalid_data(
                "decoded activation payload byte count exceeds maximum",
            ));
        }
    }
    let mut activation = vec![0; activation_bytes];
    if activation_bytes > 0 {
        reader.read_exact(&mut activation)?;
    }
    Ok(StageWireMessage {
        kind,
        pos_start,
        token_count,
        state,
        request_id,
        session_id,
        sampling,
        chat_sampling_metadata,
        tokens,
        positions,
        activation,
        raw_bytes: Vec::new(),
    })
}

fn checked_i32_len(
    value: i32,
    max: usize,
    negative_message: &'static str,
    too_large_message: &'static str,
) -> io::Result<usize> {
    if value < 0 {
        return Err(invalid_data(negative_message));
    }
    let value = usize::try_from(value).map_err(|_| invalid_data(too_large_message))?;
    if value > max {
        return Err(invalid_data(too_large_message));
    }
    Ok(value)
}

fn checked_u32_len(value: u32, max: usize, too_large_message: &'static str) -> io::Result<usize> {
    let value = usize::try_from(value).map_err(|_| invalid_data(too_large_message))?;
    if value > max {
        return Err(invalid_data(too_large_message));
    }
    Ok(value)
}

fn write_state_header(mut writer: impl Write, state: StageStateHeader) -> io::Result<()> {
    write_i32(&mut writer, state.version)?;
    write_i32(&mut writer, state.seq_id)?;
    write_i32(&mut writer, state.phase)?;
    write_i32(&mut writer, state.flags)?;
    write_i32(&mut writer, state.checkpoint_generation)?;
    write_i32(&mut writer, state.prompt_token_count)?;
    write_i32(&mut writer, state.decode_step)?;
    write_i32(&mut writer, state.current_token)?;
    write_i32(&mut writer, state.source_stage_index)?;
    write_i32(&mut writer, state.reserved)
}

fn read_state_header(mut reader: impl Read) -> io::Result<StageStateHeader> {
    Ok(StageStateHeader {
        version: read_i32(&mut reader)?,
        seq_id: read_i32(&mut reader)?,
        phase: read_i32(&mut reader)?,
        flags: read_i32(&mut reader)?,
        checkpoint_generation: read_i32(&mut reader)?,
        prompt_token_count: read_i32(&mut reader)?,
        decode_step: read_i32(&mut reader)?,
        current_token: read_i32(&mut reader)?,
        source_stage_index: read_i32(&mut reader)?,
        reserved: read_i32(&mut reader)?,
    })
}

fn write_sampling_config(mut writer: impl Write, sampling: &StageSamplingConfig) -> io::Result<()> {
    write_u32(&mut writer, sampling.flags)?;
    write_u32(&mut writer, sampling.seed)?;
    write_f32(&mut writer, sampling.temperature)?;
    write_f32(&mut writer, sampling.top_p)?;
    write_i32(&mut writer, sampling.top_k)?;
    write_f32(&mut writer, sampling.min_p)?;
    write_f32(&mut writer, sampling.presence_penalty)?;
    write_f32(&mut writer, sampling.frequency_penalty)?;
    write_f32(&mut writer, sampling.repeat_penalty)?;
    write_i32(&mut writer, sampling.penalty_last_n)?;
    let count = sampling.logit_bias.len().min(MAX_STAGE_LOGIT_BIAS);
    write_u32(&mut writer, count as u32)?;
    for bias in sampling.logit_bias.iter().take(count) {
        write_i32(&mut writer, bias.token_id)?;
        write_f32(&mut writer, bias.bias)?;
    }
    Ok(())
}

fn read_sampling_config(mut reader: impl Read) -> io::Result<StageSamplingConfig> {
    let mut sampling = StageSamplingConfig {
        flags: read_u32(&mut reader)?,
        seed: read_u32(&mut reader)?,
        temperature: read_f32(&mut reader)?,
        top_p: read_f32(&mut reader)?,
        top_k: read_i32(&mut reader)?,
        min_p: read_f32(&mut reader)?,
        presence_penalty: read_f32(&mut reader)?,
        frequency_penalty: read_f32(&mut reader)?,
        repeat_penalty: read_f32(&mut reader)?,
        penalty_last_n: read_i32(&mut reader)?,
        logit_bias: Vec::new(),
    };
    let logit_bias_count = usize::try_from(read_u32(&mut reader)?)
        .map_err(|_| invalid_data("logit bias count overflows usize"))?;
    if logit_bias_count > MAX_STAGE_LOGIT_BIAS {
        return Err(invalid_data("logit bias count exceeds maximum"));
    }
    sampling.logit_bias.reserve(logit_bias_count);
    for _ in 0..logit_bias_count {
        sampling.logit_bias.push(StageLogitBias {
            token_id: read_i32(&mut reader)?,
            bias: read_f32(&mut reader)?,
        });
    }
    Ok(sampling)
}

const REPLY_STATS_FIELD_COUNT: usize = 39;
const REPLY_STATS_WIRE_BYTES: usize = REPLY_STATS_FIELD_COUNT * std::mem::size_of::<i64>();

fn write_reply_stats(mut writer: impl Write, stats: StageReplyStats) -> io::Result<()> {
    let fields = reply_stats_fields(stats);
    let mut bytes = [0_u8; REPLY_STATS_WIRE_BYTES];
    for (chunk, value) in bytes
        .chunks_exact_mut(std::mem::size_of::<i64>())
        .zip(fields)
    {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    writer.write_all(&bytes)
}

fn read_reply_stats(mut reader: impl Read) -> io::Result<StageReplyStats> {
    let mut bytes = [0_u8; REPLY_STATS_WIRE_BYTES];
    reader.read_exact(&mut bytes)?;
    let mut fields = [0_i64; REPLY_STATS_FIELD_COUNT];
    for (field, chunk) in fields
        .iter_mut()
        .zip(bytes.chunks_exact(std::mem::size_of::<i64>()))
    {
        *field = i64::from_le_bytes(chunk.try_into().expect("i64 chunk size"));
    }
    Ok(reply_stats_from_fields(fields))
}

fn reply_stats_fields(stats: StageReplyStats) -> [i64; REPLY_STATS_FIELD_COUNT] {
    [
        stats.kv_lookup_hits,
        stats.kv_lookup_misses,
        stats.kv_lookup_errors,
        stats.kv_imported_pages,
        stats.kv_imported_tokens,
        stats.kv_recorded_pages,
        stats.kv_recorded_bytes,
        stats.kv_hit_stage_mask,
        stats.kv_record_stage_mask,
        stats.checkpoint_flush_us,
        stats.checkpoint_prefill_drain_us,
        stats.checkpoint_local_us,
        stats.checkpoint_downstream_write_us,
        stats.checkpoint_downstream_wait_us,
        stats.checkpoint_total_us,
        stats.checkpoint_prefill_drained_replies,
        stats.restore_flush_us,
        stats.restore_prefill_drain_us,
        stats.restore_local_us,
        stats.restore_downstream_write_us,
        stats.restore_downstream_wait_us,
        stats.restore_total_us,
        stats.restore_prefill_drained_replies,
        stats.verify_span_compute_us,
        stats.verify_span_forward_write_us,
        stats.verify_span_downstream_wait_us,
        stats.verify_span_total_us,
        stats.verify_span_stage_count,
        stats.verify_span_request_count,
        stats.verify_span_token_count,
        stats.verify_span_max_tokens,
        stats.verify_span_checkpointed_requests,
        stats.verify_span_skip_checkpoint_requests,
        stats.prefill_edge_write_us_max,
        stats.prefill_edge_wait_us_max,
        stats.prefill_edge_total_us_max,
        stats.prefill_edge_stage_index,
        stats.prefill_edge_activation_bytes_max,
        stats.prefill_edge_observation_count,
    ]
}

fn reply_stats_from_fields(fields: [i64; REPLY_STATS_FIELD_COUNT]) -> StageReplyStats {
    StageReplyStats {
        kv_lookup_hits: fields[0],
        kv_lookup_misses: fields[1],
        kv_lookup_errors: fields[2],
        kv_imported_pages: fields[3],
        kv_imported_tokens: fields[4],
        kv_recorded_pages: fields[5],
        kv_recorded_bytes: fields[6],
        kv_hit_stage_mask: fields[7],
        kv_record_stage_mask: fields[8],
        checkpoint_flush_us: fields[9],
        checkpoint_prefill_drain_us: fields[10],
        checkpoint_local_us: fields[11],
        checkpoint_downstream_write_us: fields[12],
        checkpoint_downstream_wait_us: fields[13],
        checkpoint_total_us: fields[14],
        checkpoint_prefill_drained_replies: fields[15],
        restore_flush_us: fields[16],
        restore_prefill_drain_us: fields[17],
        restore_local_us: fields[18],
        restore_downstream_write_us: fields[19],
        restore_downstream_wait_us: fields[20],
        restore_total_us: fields[21],
        restore_prefill_drained_replies: fields[22],
        verify_span_compute_us: fields[23],
        verify_span_forward_write_us: fields[24],
        verify_span_downstream_wait_us: fields[25],
        verify_span_total_us: fields[26],
        verify_span_stage_count: fields[27],
        verify_span_request_count: fields[28],
        verify_span_token_count: fields[29],
        verify_span_max_tokens: fields[30],
        verify_span_checkpointed_requests: fields[31],
        verify_span_skip_checkpoint_requests: fields[32],
        prefill_edge_write_us_max: fields[33],
        prefill_edge_wait_us_max: fields[34],
        prefill_edge_total_us_max: fields[35],
        prefill_edge_stage_index: fields[36],
        prefill_edge_activation_bytes_max: fields[37],
        prefill_edge_observation_count: fields[38],
    }
}

fn read_i32(mut reader: impl Read) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn write_i32(mut writer: impl Write, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(mut reader: impl Read) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn write_u32(mut writer: impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_f32(mut reader: impl Read) -> io::Result<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn write_f32(mut writer: impl Write, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u64(mut reader: impl Read) -> io::Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_u64(mut writer: impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
