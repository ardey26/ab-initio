//! Step 1 of the spike: does generated chemistry reward combining at all, and
//! does the adjacent possible grow with depth or regress to the mean?

use crate::chem::*;

const KEY_PROPS: [usize; 3] = [P_NUTRI, P_HARD, P_ENERGY];
const EPS: f32 = 0.05;
const LAYER_CAP: usize = 48;
const MAX_DEPTH: usize = 4;

pub struct SeedReport {
    pub seed: u64,
    pub improving_frac: f32,
    pub improving_frac_hot: f32,
    pub best_nutri: Vec<f32>, // per depth 0..=MAX_DEPTH
    pub best_hard: Vec<f32>,
    pub new_useful: Vec<usize>, // per depth 1..=MAX_DEPTH
    pub n_known: usize,
}

fn improves(c: &mut Chemistry, id: u32, a: u32, b: u32) -> bool {
    let (pa, pb, pc) = (c.props[a as usize], c.props[b as usize], c.props[id as usize]);
    KEY_PROPS.iter().any(|&k| pc[k] > pa[k].max(pb[k]) + EPS)
}

fn score(p: &Props) -> f32 {
    KEY_PROPS.iter().map(|&k| p[k]).fold(0.0, f32::max)
}

pub fn analyze(seed: u64, n_base: usize) -> SeedReport {
    let mut c = Chemistry::generate(seed, n_base);
    let base: Vec<u32> = (0..n_base as u32).collect();

    // Pairwise improving fraction over base materials.
    let (mut imp, mut imp_hot, mut tot) = (0usize, 0usize, 0usize);
    for i in 0..n_base as u32 {
        for j in i..n_base as u32 {
            for &hot in &[false, true] {
                let t = if hot { 1.0 } else { 0.0 };
                let id = c.combine(i, j, t);
                tot += 1;
                if improves(&mut c, id, i, j) {
                    imp += 1;
                    if hot {
                        imp_hot += 1;
                    }
                }
            }
        }
    }
    let improving_frac = imp as f32 / tot as f32;
    let improving_frac_hot = imp_hot as f32 / (tot / 2) as f32;

    // Adjacent-possible growth: frontier BFS, each layer capped by score.
    let mut layers: Vec<Vec<u32>> = vec![base.clone()];
    let mut best_nutri = vec![base.iter().map(|&i| c.props[i as usize][P_NUTRI]).fold(0.0, f32::max)];
    let mut best_hard = vec![base.iter().map(|&i| c.props[i as usize][P_HARD]).fold(0.0, f32::max)];
    let mut new_useful = Vec::new();
    let mut seen: std::collections::HashSet<u32> = base.iter().copied().collect();
    for d in 1..=MAX_DEPTH {
        let newest = layers[d - 1].clone();
        let all: Vec<u32> = layers.iter().flatten().copied().collect();
        let mut found: Vec<u32> = Vec::new();
        for &x in &newest {
            for &y in &all {
                for &hot in &[false, true] {
                    let t = if hot { 1.0 } else { 0.0 };
                    let id = c.combine(x, y, t);
                    if !seen.contains(&id) && improves(&mut c, id, x, y) {
                        seen.insert(id);
                        found.push(id);
                    }
                }
            }
        }
        new_useful.push(found.len());
        found.sort_by(|&a, &b| score(&c.props[b as usize]).partial_cmp(&score(&c.props[a as usize])).unwrap().then(a.cmp(&b)));
        found.truncate(LAYER_CAP);
        let bn = found.iter().map(|&i| c.props[i as usize][P_NUTRI]).fold(best_nutri[d - 1], f32::max);
        let bh = found.iter().map(|&i| c.props[i as usize][P_HARD]).fold(best_hard[d - 1], f32::max);
        best_nutri.push(bn);
        best_hard.push(bh);
        if found.is_empty() {
            for _ in d + 1..=MAX_DEPTH {
                new_useful.push(0);
                best_nutri.push(bn);
                best_hard.push(bh);
            }
            break;
        }
        layers.push(found);
    }
    SeedReport { seed, improving_frac, improving_frac_hot, best_nutri, best_hard, new_useful, n_known: c.n_known() }
}

pub fn run(n_seeds: u64, n_base: usize) {
    println!("seed,improving_frac,improving_frac_hot,nutri_d0,nutri_d1,nutri_d2,nutri_d3,nutri_d4,hard_d0,hard_d1,hard_d2,hard_d3,hard_d4,new_d1,new_d2,new_d3,new_d4,n_known");
    let mut reports = Vec::new();
    for s in 0..n_seeds {
        let r = analyze(s, n_base);
        println!(
            "{},{:.3},{:.3},{},{},{},{}",
            r.seed,
            r.improving_frac,
            r.improving_frac_hot,
            r.best_nutri.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(","),
            r.best_hard.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(","),
            r.new_useful.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
            r.n_known
        );
        reports.push(r);
    }
    let n = reports.len() as f32;
    let mean = |f: &dyn Fn(&SeedReport) -> f32| reports.iter().map(f).sum::<f32>() / n;
    eprintln!("--- aggregate over {} seeds ---", reports.len());
    eprintln!("mean improving pair fraction: {:.3} (hot only {:.3})", mean(&|r| r.improving_frac), mean(&|r| r.improving_frac_hot));
    for d in 0..=MAX_DEPTH {
        eprintln!(
            "depth {}: mean best nutrition {:.3}, mean best hardness {:.3}{}",
            d,
            mean(&|r| r.best_nutri[d]),
            mean(&|r| r.best_hard[d]),
            if d > 0 { format!(", mean new useful artifacts {:.1}", mean(&|r| r.new_useful[d - 1] as f32)) } else { String::new() }
        );
    }
    let grows = reports.iter().filter(|r| r.best_nutri[2] > r.best_nutri[1] + 0.02 && r.best_nutri[1] > r.best_nutri[0] + 0.02).count();
    let headroom1 = reports.iter().filter(|r| r.best_nutri[1] > r.best_nutri[0] + 0.1).count();
    let tool_gate = reports.iter().filter(|r| r.best_hard[1] > r.best_hard[0] + 0.05).count();
    eprintln!("seeds where nutrition climbs at depth 1 AND 2: {}/{}", grows, reports.len());
    eprintln!("seeds with >0.1 nutrition headroom at depth 1: {}/{}", headroom1, reports.len());
    eprintln!("seeds where a depth-1 artifact is harder than any base material: {}/{}", tool_gate, reports.len());
}
