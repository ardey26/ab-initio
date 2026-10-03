use abi_core::events::ExternalEvent;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_abi-sim"))
}

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("abi-sim-{}-{}", tag, std::process::id()))
}

fn run(dir: &Path, ticks: &str, extra: &[&str]) -> Output {
    bin().args(["run", "--seed", "3", "--ticks", ticks, "--size", "64", "--pop", "200", "--window", "20", "--checkpoint-every", "25", "--out"]).arg(dir).args(extra).output().unwrap()
}

fn replay(dir: &Path, to: &str) -> Output {
    bin().args(["replay", "--run"]).arg(dir).args(["--to", to]).output().unwrap()
}

fn hash_line(out: &Output) -> String {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).lines().find(|l| l.starts_with("hash ")).expect("hash line").to_string()
}

#[test]
fn run_then_replay_reproduces_hash() {
    let dir = tmp("main");
    let live60 = hash_line(&run(&dir, "60", &[]));
    assert!(dir.join("metrics.csv").exists());
    assert!(dir.join("config.toml").exists());
    let cks: Vec<_> = std::fs::read_dir(dir.join("checkpoints")).unwrap().collect();
    assert!(cks.len() >= 3, "checkpoints at 25, 50 and end: {}", cks.len());
    assert_eq!(hash_line(&replay(&dir, "60")), live60);
    // Replay to 40 loads checkpoint 25 and must step 15 ticks to match a live 40-tick run.
    let dir40 = tmp("live40");
    let live40 = run(&dir40, "40", &[]);
    assert_eq!(hash_line(&replay(&dir, "40")), hash_line(&live40));
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_dir_all(&dir40).unwrap();
}

#[test]
fn replay_reaches_ticks_before_first_periodic_checkpoint() {
    let dir = tmp("early");
    assert!(run(&dir, "60", &[]).status.success());
    let dir10 = tmp("early10");
    let live10 = hash_line(&run(&dir10, "10", &[]));
    assert_eq!(hash_line(&replay(&dir, "10")), live10);
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_dir_all(&dir10).unwrap();
}

#[test]
fn refuses_dirty_run_dir() {
    let dir = tmp("dirty");
    assert!(run(&dir, "10", &[]).status.success());
    let second = run(&dir, "10", &[]);
    assert!(!second.status.success());
    assert!(String::from_utf8_lossy(&second.stderr).contains("not empty"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn events_are_logged_and_replay_matches_live() {
    let events = tmp("events").with_extension("jsonl");
    let evs = [
        (30u64, ExternalEvent::Temperature { cx: 5, cy: 5, radius: 3, delta: 0.6 }),
        (45u64, ExternalEvent::DropMatter { x: 2, y: 2, material: 5, mass: 3000 }),
    ];
    let text: String = evs.iter().map(|e| serde_json::to_string(e).unwrap() + "\n").collect();
    std::fs::write(&events, text).unwrap();
    let ev = events.to_str().unwrap();

    let dir = tmp("ev60");
    let live60 = hash_line(&run(&dir, "60", &["--events", ev]));
    let log = std::fs::read_to_string(dir.join("events.log")).unwrap();
    assert_eq!(log.lines().count(), 2);
    assert_eq!(hash_line(&replay(&dir, "60")), live60);

    let dir50 = tmp("ev50");
    let live50 = hash_line(&run(&dir50, "50", &["--events", ev]));
    assert_eq!(hash_line(&replay(&dir, "50")), live50);

    // Replay to 46 loads checkpoint 25 and steps across both events. The tick-45 event fires
    // during the step from 45 to 46, so 46 is the first tick that crosses it.
    let dir46 = tmp("ev46");
    let live46 = hash_line(&run(&dir46, "46", &["--events", ev]));
    assert_eq!(hash_line(&replay(&dir, "46")), live46);

    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_dir_all(&dir50).unwrap();
    std::fs::remove_dir_all(&dir46).unwrap();
    std::fs::remove_file(&events).unwrap();
}

#[test]
fn bad_event_line_reports_line_number() {
    let events = tmp("badev").with_extension("jsonl");
    std::fs::write(&events, "[30,{\"Impact\":{\"cx\":1,\"cy\":1,\"radius\":1}}]\nnot json\n").unwrap();
    let dir = tmp("badev");
    let out = run(&dir, "10", &["--events", events.to_str().unwrap()]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("events file line 2"));
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_file(&events).unwrap();
}

#[test]
fn zero_window_is_rejected() {
    let dir = tmp("zero");
    let out = run(&dir, "10", &["--window", "0"]);
    assert!(!out.status.success());
    assert!(!dir.exists());
}

#[test]
fn verify_reports_deterministic() {
    let out = bin().args(["verify", "--seed", "4", "--ticks", "40", "--size", "64", "--pop", "150"]).output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("DETERMINISTIC"));
}
