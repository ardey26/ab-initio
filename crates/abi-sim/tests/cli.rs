use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_abi-sim"))
}

#[test]
fn run_then_replay_reproduces_hash() {
    let dir = std::env::temp_dir().join(format!("abi-sim-{}", std::process::id()));
    let out = bin().args(["run", "--seed", "3", "--ticks", "60", "--size", "64", "--pop", "200", "--window", "20", "--checkpoint-every", "25", "--out"]).arg(&dir).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let hash_line = stdout.lines().find(|l| l.starts_with("hash ")).expect("hash line");
    assert!(dir.join("metrics.csv").exists());
    assert!(dir.join("config.toml").exists());
    let cks: Vec<_> = std::fs::read_dir(dir.join("checkpoints")).unwrap().collect();
    assert!(cks.len() >= 3, "checkpoints at 25, 50 and end: {}", cks.len());
    let replay = bin().args(["replay", "--run"]).arg(&dir).args(["--to", "60"]).output().unwrap();
    assert!(replay.status.success(), "{}", String::from_utf8_lossy(&replay.stderr));
    let rs = String::from_utf8_lossy(&replay.stdout);
    assert_eq!(rs.lines().find(|l| l.starts_with("hash ")).unwrap(), hash_line);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn verify_reports_deterministic() {
    let out = bin().args(["verify", "--seed", "4", "--ticks", "40", "--size", "64", "--pop", "150"]).output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("DETERMINISTIC"));
}
