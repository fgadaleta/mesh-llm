mod cli;
mod native_mtp_openai;
mod report;
mod runner;
mod support;

use anyhow::Result;
use clap::Parser;

use crate::{
    cli::{Cli, CommandKind},
    native_mtp_openai::native_mtp_openai_ab,
    runner::{chain, dtype_matrix, single_step, split_scan, state_handoff},
};

fn main() -> Result<()> {
    match Cli::parse().command {
        CommandKind::SingleStep(args) => single_step(args),
        CommandKind::Chain(args) => chain(args),
        CommandKind::SplitScan(args) => split_scan(args),
        CommandKind::DtypeMatrix(args) => dtype_matrix(args),
        CommandKind::StateHandoff(args) => state_handoff(args),
        CommandKind::NativeMtpOpenAiAb(args) => native_mtp_openai_ab(*args),
    }
}
